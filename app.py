import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys
from focus_model import *
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from plot import *


# check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# DATASET
dataset = 'modanet'

# PARAMETERS
yolo_params =  {   
  "model_def" : "yolo/modanetcfg/yolov3-modanet.cfg",
  "weights_path" : "yolo/weights/yolov3-modanet_last.weights",
  "class_path":"yolo/modanetcfg/modanet.names",
  "conf_thres" : 0.5,
  "nms_thres" :0.4,
  "img_size" : 416,
  "device" : device
}

#Classes
classes = load_classes(yolo_params['class_path'])

#Colors
cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])

# MODEL
model = 'yolo'
detectron = YOLOv3Predictor(params=yolo_params)

def input_image():
  
  while True:
    path = input('img path: ')
    # path = 'tests/harry1.jpeg'
    img_name = path.split('/')[-1]
    if not os.path.exists(path):
      print('Img does not exists..')
    else:
      break
  
  return path,img_name

def single_image_process():
  # read image_path
  path,img_name = input_image()

  # focus the image by blurring the background
  resized_im, seg_map = run_visualization(path)
  img, binary_mapping_img = focus_person(path,resized_im,seg_map)
  # cv2.imwrite('temp3.jpg',img)

  # obtain detections on the modified image
  detections = detectron.get_detections(img)
  plot_detections(detections,img,img_name,classes,colors)

def batch_image_process():

  # read folder_path
  folder_path = input('Enter the folder path: ')

  # iterate over all images in the folder
  for img_name in os.listdir(folder_path):

    # focus the image by blurring the background
    path = os.path.join(folder_path,img_name)
    resized_im, seg_map = run_visualization(path)
    img, binary_mapping_img = focus_person(path,resized_im,seg_map)
    # cv2.imwrite('temp3.jpg',img)

    # obtain detections on the modified image
    detections = detectron.get_detections(img)
    plot_detections(detections,img,img_name,classes,colors)



# operational mode
# batch_image_process()
single_image_process()
