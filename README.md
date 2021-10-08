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
python main.py -i test\0.jpg -d yolo
```
- `-i`: path to input image to be LP-detected.
- `-d`: detection model (wpod or yolo). This project uses WPOD-Net as default model for license plate detection. The detailed model architecture is presented [here](https://paperswithcode.com/paper/license-plate-detection-and-recognition-in)

## YOLOv4 detection model 
Although WPOD-Net is very good at detecting LPs and transform them to a direct viewpoint, its performance on square license plates is still limited. Therefore, I add a custom YOLOv4 tiny pretrained model for square license plates only. You can take a look at my [colab notebook](https://colab.research.google.com/drive/1L2E8j45KTyyv0PcF-3hI08ZzWFtAn_lf) or step-by-step tutorial here: https://nttuan8.com/bai-toan-phat-hien-bien-so-xe-may-viet-nam/

## Disadvantages
- WPOD-Net couldn't detect some square license plates.
- Text recognition performance affected by brightness, blur, and perspective of license plate.

## References
1. Detection phase:
- https://www.miai.vn/2019/11/20/nhan-dien-bien-so-xe-chuong-2-phat-hien-bien-so-xe-bang-pretrain-wpod-net/
- https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-1-detection-795fda47e922
2. Text recognition phase:
- https://www.pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/
