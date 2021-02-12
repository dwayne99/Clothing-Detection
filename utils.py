import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os

###############################################################################################
############ DEEPLAB FOR PERSON SEGMENTATION ALONG WITH FUNCTIONS FOR VISUALIZATION ###########

def load_model():
    """
    Load the DeepLabV3 model for instance segmentation. uses the resnet101 as its backbone
    """
    
    # Load the DeepLab v3 model to system
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.to(device).eval()
    return model

def get_pred(img, model):
    """
    PARAMS:
        img: find detections in img
        model: model used for the evaluation
    OUTPUT:
        output : segmented image
    """
    
    # See if GPU is available and if yes, use it
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the standard transforms that need to be done at inference time
    imagenet_stats = [[0.485, 0.456, 0.406], [0.485, 0.456, 0.406]]
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = imagenet_stats[0],std  = imagenet_stats[1])
    ])
    # All pre-trained models expect input images in mini-batches of 3-channel RGB images
    # of shape (N, 3, H, W). Our image is (3, H, W) so we have to unsqueeze to get (1, 3, H, W)
    input_tensor = preprocess(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Make the predictions for labels across the image
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output = output.argmax(0)

    # Return the predictions
    return output.cpu().numpy()

def blur_background(img, seg_img):
    """
    PARAMS:
        img: image to be focused 
        seg_img: semented image obtained from deeplab model
    OUTPUT:
        blur_img: person focused and background blurred image
    """
    # get the dimentions
    width, height, channels = img.shape
    # Define the kernel size for applying Gaussian Blur
    blur_value = (21, 21)
    # Wherever there's empty space/no person, the label is zero 
    # Hence identify such areas and create a mask (replicate it across RGB channels)
    mask = seg_img != 15
    mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

    # Apply the Gaussian blur for background with the kernel size specified in constants above
    blur = cv2.GaussianBlur(img, blur_value, 0)
    img[mask] = blur[mask]
    
    return img, mask

def plot_blur_mask(img1,img2):
    """
    PARAMS:
        img1: Focused image with blur
        img2: Binary Mask of the person instance segmentation
    OUTPUT:
        Visualization of the two
    """    
    # sublot for blurred background image
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.xticks([]);plt.yticks([])
    plt.title('Blurred Image')

    # sublot for mask
    plt.subplot(1, 2, 2)
    plt.imshow(img2*255)
    plt.xticks([]);plt.yticks([])
    plt.title('Mask')
    plt.show()
    
###############################################################################################

############################## USEFUL FUNCTIONS ###############################################

def input_image():
    """
    OUTPUT:
        path: path of the image if it exists
        img_name: name of the image (with extension)
    """
    while True:
        path = input('img path: ')
        # path = 'tests/harry1.jpeg'
        img_name = path.split('/')[-1]
        if not os.path.exists(path):
            print('Img does not exists..')
        else:
            break
    return path,img_name

###############################################################################################

########################## DETECTRON DETECTIONS ###############################################

def plot_clothing_detections(detections, img, img_name, classes, colors, OUT_PATH):
    """
    Functions to plot and save the results of the clothing detections
    
    PARAMS:
        detections: detections of the clothing items
        img: input image
        img_name: input image name
        classes: list of clothing classes 
        colors: list of colors for the bounding boxes
        OUT_PATH: directory path to save all the results
    OUTPUT:
        A saved image with the plottings of the bounding boxes at the desired location
    """
    if len(detections) != 0 :
#         result = ""
        detections.sort(reverse=False ,key = lambda x:x[4])
        
        for x1, y1, x2, y2, cls_conf, cls_pred in detections:
            """
            x1,y1 : top left co-ordinates of bounding box
            x2,y2 : bottom right co-ordinates of bounding box
            cls_conf : class confidence score
            cls_pred : class of the object
            """
            
            # round the coordinates to int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
#             result += "\t+ Label: %s, Conf: %.5f\n" % (classes[int(cls_pred)], cls_conf)         

            color = colors[int(cls_pred)]
            color = tuple(c*255 for c in color)
            color = (.7*color[2],.7*color[1],.7*color[0])       
            font = cv2.FONT_HERSHEY_SIMPLEX   
            text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)
            cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
            y1 = 0 if y1<0 else y1
            y1_rect = y1-25
            y1_text = y1-5

            if y1_rect<0:
                y1_rect = y1+27
                y1_text = y1+20
            cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
            cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        img_name = img_name.split('.')[0]
        cv2.imwrite(OUT_PATH + '/' + img_name + '.png',img)
        print(f'Saved successfully at {OUT_PATH}')
        
    else:
        print('No detections were found in the image...')