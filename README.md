# Environment
conda create -n MDDCNet python=3.10 \
conda activate MDDCNet \
pip install torch==2.1.1 torchvision torchaudio 

# Prepare Dataset

<pre>
├── Dataset
    ├── images
    │   ├── train
    │   │   └── 00001.png
    │   ├── val
    │   └── test
    └── labels
        ├── train
        │   └── 00001.txt
        ├── val
        └── test
</pre>


# Training

python train.py --task train --data /ultralytics/cfg/datasets/KITTI.yaml \
--config  /ultralytics/cfg/models/mamba-yolo/MDDCNet-T.yaml  \
--amp  --project ./output_dir --name TrainResult

