# Face Mask Detection 
Real time face mask detection using Python, OpenCV and Keras.

## Demonstration of Project:
<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/face_mask_detected.gif?raw=true" />
</p>

## Table of Contents:
1. [Abstract](https://github.com/parvanehyaghoubi/Face_Mask_Detection#1-abstract)
2. [Dataset Creation](https://github.com/parvanehyaghoubi/Face_Mask_Detection#2-dataset-creation)
3. [Requirements](https://github.com/parvanehyaghoubi/Face_Mask_Detection#3-requirements)
4. [Results](https://github.com/parvanehyaghoubi/Face_Mask_Detection#4-results)
5. [How to Use](https://github.com/parvanehyaghoubi/Face_Mask_Detection#5-how-to-use)

## 1. Abstract
- This project is concerned with the detection of face masks.
- There are tree operation in this project:
  - Detect face mask on images (using Keras/TensorFlow)
  - Detect face mask on real  time video stream and webcam
  - Also, you can find your file in file dialog or you can run webcam.
- Dataset consist of  7,553 images combined manually.
- Adam is used as optimizer and Categorical Crossentropy is used as a loss function.

<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/with_mask_detection.png?raw=true" />
</p>
<p align=center> 
Figure 1: With Mask
</p>

<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/without_mask_detection.png?raw=true" />
</p>

<p align=center> 
Figure 2: Without Mask
</p>

## 2. Dataset Creation
### Link to Download Complete Dataset, data.npy and target.npy to preprocessing data:
[Datasets](https://drive.google.com/drive/folders/1o4iNmnXvIAsR2RArvRP-prliNfbdVbTr?usp=sharing)

## 3. Requirements
- Jupyter Notebook version 6.1 or above
- Python version 3.8 or above
- Python Libraries Used:
  - numpy https://numpy.org/doc/
  - pandas https://pandas.pydata.org/docs/
  - TensorFlow https://www.tensorflow.org/api_docs
  - OpenCV https://docs.opencv.org/3.4/
  - scikit-learn https://scikit-learn.org/stable/user_guide.html
  - keras https://keras.io/guides/
  - matplotlib https://matplotlib.org/stable/users/index.html
  - tkinter https://docs.python.org/3/library/tkinter.html


 ## 4. Results
The plots of Model Accuracy and Model Loss are as follows:

<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/Model_Accuracy.png?raw=true" />
</p>


<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/Model_Loss.png?raw=true" />
</p>

## 5. How to Use

To use this project on your system, follow these steps:

1. Clone this repository onto your system by typing the following command on your Command Prompt:
```
git clone https://github.com/parvanehyaghoubi/Face_Mask_Detection
```

2. Download all libaries using:
```
pip install -r requirements.txt
```

3. Run the application:
```
python main_mask_detector.py
```

4. You can see an environment which create with tkinter:

<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/Root_Tkinter.png?raw=true" />
</p>

5. If you choose `Open a File`, the file dialog opens and you can choose your video

<p align="center">
  <img src="https://github.com/parvanehyaghoubi/Face_Mask_Detection/blob/main/images/File_Dialog.png?raw=true" />
</p>

6. If you choose `Open Camera`, you can use your camera to face mask detection!!

#### The Project is now ready to use !!

#### Contact
For any inquiries or feedback, please contact:
- parvaneh.yaghoubi77@gmail.com

#### Link
##### Parvaneh Yaghoubi
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/parvaneh-yaghoubi-54362620b/)

