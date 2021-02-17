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

def blur_background(img, mask):
    """
    PARAMS:
        img: image to be focused 
        mask: person mask for the image
    OUTPUT:
        out: person focused and background blurred image
    """
    mask = mask!=1
    blur = cv2.blur(img,(21,21),0)
    out = img.copy()
    out[mask>0] = blur[mask>0]
    
    return out,mask
    

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
    plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
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
############################### FUNCTION TO FOCUS THE IMAGE ###################################
def focus_with_distance(img, bw):
    """
    PARAMS:
        img: image to be focused (np array)
        bw: binary mask of img (np array)
    OUTPUT:
        focused_img: img with the distance transform applied
    """
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
    dist[dist < 0.01] =  0.005
    dist[(dist < 0.9) & (dist > 0.95) ] =  0.9
    dist[(dist < 1) & (dist > 0.95) ] =  0.95
    dist = -np.log(dist)
    cv2.normalize(dist, dist, 0, 1, cv2.NORM_MINMAX)
    
    # applying the layers
    focused_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(focused_img)
    lim = 0
    #s[binary_mapping_img == 0] = 30
#     print("shapes of dist and s", dist.shape, s.shape)
    #s_new = cv2.multiply(s,1-dist, cv2.CV_8U)
    s_new = np.uint8(np.multiply(s, dist))
    v_new = np.uint8(np.multiply(v, dist))
    focused_img = cv2.merge((h, s_new, v_new))
    focused_img = cv2.cvtColor(focused_img, cv2.COLOR_HSV2BGR)
    
    return focused_img

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
        result = ""
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
            
            result += "\t+ Label: %s, Conf: %.5f\n" % (classes[int(cls_pred)], cls_conf)         
            result += "\t (x1,y1,x2,y2) = (%s, %s, %s, %s)\n" % (x1,y1,x2,y2)         

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
        
        cv2.imwrite(OUT_PATH + '/' + img_name + '.png',img)
        print(f'Saved successfully at {OUT_PATH}')
        
        with open(OUT_PATH+'/'+img_name+'.txt', 'a') as f:
            f.write(result)
        
        print(result)
    else:
        print('No detections were found in the image...')
        
        
def crop_garments(img,detections,dest_pth,classes,img_name):
    """
    PARAMS:
        img: processed image
        detections: detections of the garments obtained from detectron
        dest_path: destination of the directory to save the cropped images
        classes: list of the fashion items 
        img_name: name of the image
    OUPUT:
        save the individual crops of every garment
    """
    # categories of the fashion items
    cat1 = [ 'outer', 'dress', 'pants', 'top', 'shorts', 'skirt']
    cat2 = ['bag', 'belt', 'boots','footwear','sunglasses','headwear', 'scarf/tie']
    
    for i, (x1, y1, x2, y2, cls_conf, cls_pred ) in enumerate(detections,1):
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = classes[int(cls_pred)]
        
        desired_size = 300 if label in cat1 else 150

        im = img[y1:y2,x1:x2]
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        
        try:
            os.mkdir(dest_pth+'/'+img_name)
        except:
            pass
        cv2.imwrite(dest_pth+'/'+img_name+'/'+str(i)+'_'+label+'.png', new_im)
    