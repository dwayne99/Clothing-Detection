import torch
import os
import cv2
from yolo.utils.utils import *
from predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys
# from focus_model import *
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from plot import *

from utils import *
from pdb import set_trace
from instance_seg import get_prediction


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



def single_image_process():
    """
    Single-image operational mode
    """
    # read image_path
    IMG_PATH, img_name = input_image()
    img_name = img_name.split('.')[0]
    OUT_PATH = input('Enter the path to the directory to save the file: ') 
    img = cv2.imread(IMG_PATH)
    
    # focus the image by blurring the background and applying distance-transform
#     model = load_model()
#     seg = get_pred(img,model)
    masks, boxes, pred_cls = get_prediction(img, 0.5)
    for i in range(len(masks)):
        mask = masks[i]!=1
        
        # blur the image with mask
        blur = cv2.blur(img,(21,21),0)
        out = img.copy()
        out[mask>0] = blur[mask>0]
        
#         img_blur,mask = blur_background(img,seg)
#         mask = cv2.cvtColor(np.uint8(mask)*255,cv2.COLOR_BGR2GRAY)
        img_blur = focus_with_distance(out,np.uint8(mask))
    # obtain detections on the modified image
#     IMG_PATH, img_name = 'temp/1.png', '1.png'
#     img_blur = cv2.imread(IMG_PATH)
        detections = detectron.get_detections(img_blur)
#     plot_clothing_detections(detections, img_blur, img_name, classes, colors, OUT_PATH)
        crop_garments(img_blur,detections, OUT_PATH,classes,img_name)
    print('Processed successfully!..')

def batch_image_process():
    """
    Multi-image operational mode
    """
    model = load_model()
    
    # read folder_path
    folder_path = input('Enter the folder path: ')
    OUT_PATH = input('Enter the path to the directory to save the file: ') 
    # iterate over all images in the folder
    for img_name in tqdm(os.listdir(folder_path)):
        
        try:
            # read image_path
            IMG_PATH = os.path.join(folder_path,img_name)
            img = cv2.imread(IMG_PATH)

            # focus the image by blurring the background and distance transform
            seg = get_pred(img,model)
            img_blur,mask = blur_background(img,seg)
            mask = cv2.cvtColor(np.uint8(mask)*255,cv2.COLOR_BGR2GRAY)
            img_blur = focus_with_distance(img_blur,mask)
            
            # obtain detections on the modified image
            detections = detectron.get_detections(img_blur)
#             plot_clothing_detections(detections, img_blur, img_name, classes, colors, OUT_PATH)
            
            img_name = img_name.split('.')[0]
            crop_garments(img_blur,detections, OUT_PATH,classes,img_name)
        except:
            print(f'Error occured with image : {IMG_PATH}')

    print('Processed successfully!..')


# operational mode
# batch_image_process()
single_image_process()
