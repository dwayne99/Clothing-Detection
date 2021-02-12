import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from constants import *
from PIL import Image, ImageFilter, ImageDraw
from random import randint

def circular_focus(img,x1,y1,x2,y2):
    # Open the input image as numpy array, convert to RGB
    img = Image.fromarray(np.uint8(img)).convert('RGB')
    npImage=np.array(img)
    h,w=img.size
    max_hw = max(h,w)

    # Create same size alpha layer with circle
    alpha = Image.new('L', img.size,0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([x1,y1,x2,y2],0,360,fill=255)

    # Convert alpha Image to numpy array
    npAlpha=np.array(alpha)

    # Add alpha layer to RGB
    npImage=np.dstack((npImage,npAlpha))
    buff = 10
    npImage = npImage[y1-buff:y2+buff, x1-buff:x2+buff]

    # Save with alpha
    return  Image.fromarray(npImage)#.convert('RGB')

def focus2(img,x1,y1,x2,y2):

    blurred_original_image = cv2.medianBlur(img,9)
    crop = img[y1:y2,x1:x2]
    blurred_original_image[y1:y2,x1:x2] = crop
    cir_img = circular_focus(img,x1,y1,x2,y2)
    # cv2.imwrite('output3/harry1',cir_img)
    return cir_img


def plot_detections(detections,img,img_name,classes,colors):

    if len(detections) != 0 :
        detections.sort(reverse=False ,key = lambda x:x[4])
        i =1
        name = img_name.split('.')[0]
    try:
        os.mkdir('output3/'+ name)
    except:
        pass

    file_name = 'output3/'+ name + '/results.txt'
    f = open(file_name,'a')

    for x1, y1, x2, y2, cls_conf, cls_pred in detections:

        f.write("\t+ Label: %s, Conf: %.5f\n" % (classes[int(cls_pred)], cls_conf))           

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        img_focus = focus2(img,x1,y1,x2,y2)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_img = 'output3/' + name +'/' + str(i) + '.png'  
        #img_focus.save(out_img)
        cv2.imwrite(out_img,np.float32(img_focus))
        i+=1

