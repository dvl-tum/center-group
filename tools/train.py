
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint as _ModelCheckpoint
import pytorch_lightning as pl

from mmcv import Config, DictAction
from mmcv.parallel import collate
from mmpose.models import build_posenet
from mmpose.datasets import build_dataset, build_dataloader

from torch.utils.data import DataLoader
import numpy as np

import os
import os.path as osp
from functools import partial
from datetime import datetime
import argparse
import glob

import centergroup

class ModelCheckpoint(_ModelCheckpoint):
    # PL will not save every epoch otherwise!!!!!!
    def on_train_epoch_end(self, trainer, pl_module):
        self.save_checkpoint(trainer)

class MMPoseData(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
    def train_dataloader(self):
        dataset = build_dataset(self.data_cfg['train'])
        dataloader = DataLoader(
                            dataset,
                            batch_size=self.data_cfg['samples_per_gpu'],
                            num_workers=self.data_cfg['workers_per_gpu'],
                            collate_fn=partial(collate, samples_per_gpu=self.data_cfg['samples_per_gpu']),
                            #collate_fn=collate,
                            pin_memory=True,
                            shuffle=True)

        return dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('--cfg', help='Config file path')
    parser.add_argument('--out', help='File path used to save checkpoints and logs', default='output')
    parser.add_argument('--run_str', default='dflt_training')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()

    return args

def resume_training_maybe(out, cfg, run_str):
    """Check if there's already any training with the given run_str, and recover its last checkpoint"""
    path_to_search = osp.join(out, osp.basename(cfg).split('.')[0], run_str)
    candidates = glob.glob(path_to_search+'*')

    if len(candidates) > 0:
        resume_dir = max(candidates)
    
        ckpt_path = osp.join(resume_dir, 'checkpoints', 'last.ckpt')
        if osp.exists(ckpt_path):
            version = osp.basename(resume_dir)
            print(f"RESUMING TRAINING FROM {ckpt_path}")
            return version, ckpt_path
        
    print(f"TRAINING FROM SCRATCH")
    ckpt_path=None
    version = '_'.join((run_str, datetime.now().strftime("%m-%d-%Y_%H-%M-%S")))

    return version, ckpt_path

def main():
    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if not osp.exists(args.out):
        os.makedirs(args.out)

    # Autoscale LR: LR specified is for training on a single GPU with batch size 42, scale it with linear rule
    bs = cfg['data']['samples_per_gpu']
    cfg['model']['train_cfg']['optimizer_']['lr'] *= (bs/42.) * args.num_gpus
    #cfg['model']['train_cfg']['optimizer_']['bu_lr'] *= (bs/42.) * args.num_gpus


    # Try to automatically resume training:
    version, ckpt_path = resume_training_maybe(args.out, args.cfg, args.run_str)

    model = build_posenet(cfg['model'])
    datamodule = MMPoseData(cfg['data'])
    trainer = pl.Trainer(gpus= args.num_gpus,
                         plugins=DDPPlugin(find_unused_parameters=False),
                         min_epochs=100,
                         accelerator='ddp',
                         #limit_train_batches=0.001,
                         gradient_clip_val=1.0,
                         precision = 16,                        
                         callbacks = [ModelCheckpoint(monitor='loss/loc_loss/train', mode='min', save_top_k=-1, verbose=True, save_last=True)],
                         logger = pl.loggers.TensorBoardLogger(args.out, 
                                                               name= osp.basename(args.cfg).split('.')[0], 
                                                               version=version))

    trainer.fit(model, datamodule, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()