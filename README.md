# ALPR-Project

 This is an **ALPR (Auto License Plate Recognition)** project for both *rectangle* and *square* license plates.

## Setups
1. Clone the repository:
```
 git clone https://github.com/DoDucNhan/ALPR-Project.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run project
Run the following command.
```
python main.py -i test\0.jpg -d detection\wpod-net.json -e enhancement\models\EDSR_x4.pb
```
- `-i`: path to input image to be LP-detected.
- `-d`: path to detection model. This project uses WPOD-Net for license plate detection. The detailed model architecture is presented [here](https://paperswithcode.com/paper/license-plate-detection-and-recognition-in)
- `-e`: path to enhance resolution model. There are 3 different models to enhance the resolution of the LP-region detected, this might change the performace with corresponding input image.

## References
1. Detection phase:
- https://www.miai.vn/2019/11/20/nhan-dien-bien-so-xe-chuong-2-phat-hien-bien-so-xe-bang-pretrain-wpod-net/
- https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922
2. Enhanace resolution phase:
- https://www.pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/
3. Text recognition phase:
- https://www.pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
