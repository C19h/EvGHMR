## Human Mesh Reconstruction via Complementary Learning from Event Streams and Frames
Liaogehao Chen, Zhenjun Zhang, and Yaonan Wang. TMM 2026

## Preparation
For the preparation of the dataset and environment, please refer to https://github.com/JimmyZou/EventHPE

Download the pre-trained DINOv2 model dinov2_vits14_pretrain.pth from [pre-trained model](https://drive.google.com/drive/folders/1blvrnbZP3IPNOOYhQzdz5Ml4z-vnKBb9?usp=drive_link), and place it in the `./pretrained` folder
## Training and Evaluation
```
cd event_pose_estimation
python train.py   # For training
python test.py    # For evaluation
```
## Acknowledgements

We would like to thank the creators of the following excellent resources, who have provided support for our work:

EventHPE inspired our approach and provided a valuable baseline and dataset.
GitHub: https://github.com/JimmyZou/EventHPE

EventPointMesh provided a valuable baseline and dataset.
GitHub: https://github.com/RyosukeHori/EventPointMesh