# Installation
1. Clone and enter this repository:
    ```
    git clone https://github.com/dvl-tum/center-group
    cd center-group
    ```
2. (Recommended) Create a conda environment for this repo:
    ```
    conda create -n centergroup python=3.8.5 -y
    conda activate centergroup
    ```
3. Install [PyTorch](https://pytorch.org/):
    ```
    conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.1 -c pytorch
    ```
    These are the versions we used. Newer ones may also work. Make sure your system's CUDA version matches the one you specify for `cudatoolkit` here.
4. Install [mm-cv](https://github.com/open-mmlab/mmcv):
    ```
    pip install mmcv-full==1.3.10 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.1/index.html
    ```
    Make sure that you replace `cu101` for the CUDA version that you used to install PyTorch, if you used a different one. 
5. Install [this version of mmpose](https://github.com/open-mmlab/mmpose/tree/65eb45b75da0ca48812f5398a3955c75683e37b5), and all of its requirements. To do so, you can run the following:
    ```
    git clone git@github.com:open-mmlab/mmpose.git
    cd mmpose
    git checkout 65eb45b
    pip install -r requirements.txt
    python setup.py develop
    ```
6. Data preparation. Please follow the instructions [here](https://github.com/open-mmlab/mmpose/blob/65eb45b75da0ca48812f5398a3955c75683e37b5/docs/tasks/2d_body_keypoint.md) to download the [COCO Keypoints](https://cocodataset.org/#home) and [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) datasets in a directory of your choice. You don't need to download the person detection results. Please download the datasets under `$CENTERGROUP_ROOT/data`, or update the entry `data_root` in the config files if you store them somewhere else.

7. Download our trained models in `$CENTERGROUP_ROOT/models`, you can find them [here](./MODEL_ZOO.md)