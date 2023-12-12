# clone YOLOv5 and 
!git clone https://github.com/ultralytics/yolov5 
%cd yolov5
%pip install -qr requirements.txt  # installing dependencies

import torch
import os 
from IPython.display import Image, clear_output # to display images 

# if your using google colab the you can use this line of code
# set up environment 
os.environ["DATASET_DIRECTORY"]="/content/datasets"


!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="BUQIxo9xwUYVHrkPx9Ik")
project = rf.workspace("nirma-university-1bjp1").project("potholes-kedu9")
dataset = project.version(4).download("yolov5")


# training the model of the yolov5
!python train.py --img 416 --batch 16   --epochs 100 --data {dataset.location}/data.yml --weights yolov5s.pt --cache


!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/valid/images 

#display all the test images 

import glob 
from IPython.display import Images, diplay 

i=0;
#choose the correct exp folder -see prev output block 
      for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
        i+=1

      if i<30: # here number is the images that valid in for testing 
        display(Image(filename=imageName))
        print("\n")

# if your using google colab the you can use this line of code
# save model 
# export your model's weigths for future use 
    from google.colab import files
    files.download('./runs/train/exp/weights/best.pt')
