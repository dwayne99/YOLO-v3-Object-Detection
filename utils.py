from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def get_test_input(img,inp_dim):
    # img = cv2.imread("filename.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    ARGS: prediction (our output), inp_dim (input image dimension),
        anchors, num_classes, and an optional CUDA flag

    DESC:predict_transform function takes a detection feature map and turns it 
        into a 2-D tensor, where each row of the tensor corresponds to attributes
        of a bounding box

        Another problem is that since detections happen at three scales, the 
        dimensions of the prediction maps will be different. Although the 
        dimensions of the three feature maps are different, the output processing
        operations to be done on them are similar. It would be nice to have to do
        these operations on a single tensor, rather than three separate tensors.
        To remedy these problems, we introduce the function predict_transform
    '''
    #predictions.size() = (B x 3*85 x  grid x grid)

    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    # grid_size = prediction.size(2)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    #The dimensions of the anchors are in accordance to the height and width 
    #attributes of the net block. These attributes describe the dimensions of the 
    #input image, which is larger (by a factor of stride) than the detection map. 
    #Therefore, we must divide the anchors by the stride of the detection feature map.
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0]) #center-x
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1]) #center-y
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4]) #object confidence

    # add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    # add the offsets to get x, y w.r.t the grids
    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    #Apply sigmoid activation to the the class scores
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # The last thing we want to do here, is to resize the detections map to the 
    # size of the input image. The bounding box attributes here are sized according
    # to the feature map (say, 13 x 13). If the input image was 416 x 416, we 
    # multiply the attributes by 32, or the stride variable.
    prediction[:,:,:4] *= stride

    return prediction

# Testing the unique function
def unique(tensor):
    # convert tensor to numpy
    tensor_np = tensor.cpu().numpy()
    # find unique
    unique_np = np.unique(tensor_np)
    # create a tenor of unique_np
    unique_tensor = torch.from_numpy(unique_np)
    # the above is simply two instances pointing to a single tensor
    # we create an independent tensor of same type as input tensor and shape of unique tensor
    tensor_res = tensor.new(unique_tensor.shape)
    # we copy values of the unique tensor to this independent tensor (tensor_res)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def bbox_iou(box1, box2):
    '''Returns the IoU of two bounding boxes'''

    # get the coordinates of the bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the coordinates of the intersection of the rectangles
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    '''
    >>> a = torch.randn(4)
    >>> a
    tensor([-0.0299, -2.3184,  2.1593, -0.8883])
    >>> torch.clamp(a, min=0.5)
    tensor([ 0.5000,  0.5000,  2.1593,  0.5000])
    '''
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def write_results(prediction,confidence,num_classes,nms_conf = 0.4):
    '''
    ARGS : The functions takes as as input the prediction, confidence 
           (objectness score threshold), num_classes (80, in our case) and 
           nms_conf (the NMS IoU threshold).
    '''

    # STEP 1: ELIMINATE ALL THE PREDICTIONS WHOSE OBJECT CONFIDENCE SCORE IS LESS THAN THAT OF THE CONFIDENCE THRESHOLD
    #Object Confidence Thresholding. Our prediction tensor contains information 
    #about B x 10647 bounding boxes. For each of the bounding box having a objectness 
    #score below a threshold, we set the values of it's every attribute 
    #(entire row representing the bounding box) to zero.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # STEP 2: NON-MAXIMAL SUPRESSION
    # However, it's easier to calculate IoU of two boxes, using coordinates of a 
    #pair of diagnal corners of each box. So, we transform the (center x, center y, height, width) 
    #attributes of our boxes, to (top-left corner x, top-left corner y, right-bottom corner x, right-bottom corner y).
    # note: 0=x, 1=y, 2=w, 3=h
    box_corner = prediction.new(prediction.shape)
    #top left x & y
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    #bottom right x & y
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    # write flag is used to indicate if we have or haven't initialized output
    write = False

    for ind in range(batch_size):

        #image tensor 2D
        image_pred = prediction[ind]

        #each bounding box row has 85 attributes, out of which 80 are the class 
        #scores. At this point, we're only concerned with the class score having 
        #the maximum value. So, we remove the 80 class scores from each row, and 
        #instead add the index of the class having the maximum values, as well 
        #the class score of that class.
        
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+num_classes],1)  
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5],max_conf,max_conf_score)
        image_pred = torch.cat(seq,1)

        non_zero_ind = (torch.nonzero(image_pred[:,4]))

        #The try-except block is there to handle situations where we get no 
        #detections. In that case, we use continue to skip the rest of the loop body for this image.
        try: 
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        #For PyTorch 0.4 compatibility
        #Since the above code with not raise exception for no detection 
        #as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue 

        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class label
        #img_classes (ex: tensor([5, 6, 7, 8, 9]) )

        # perform nms for each class at a time
        for cls in img_classes:

            # get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            #sort the detections such that the entry witht the maximum object confidence
            #is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4],descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0) # first detection with highest confidence

            for i in range(idx):
                # get the IOU's of all boxes that come after the one we are looking
                # at in the loop


                #we have put the line of code to compute the ious in a try-catch block. 
                #This is because the loop is designed to run idx iterations (number of rows in image_pred_class).
                # However, as we proceed with the loop, a number of bounding boxes may be removed from image_pred_class. 
                #This means, even if one value is removed from image_pred_class, we cannot have idx iterations. 
                #Hence, we might try to index a value that is out of bounds (IndexError), 
                #or the slice image_pred_class[i+1:] may return an empty tensor, assigning which triggers a ValueError.
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0),1).fill_(ind)

            #repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
                
    try:
        return output
    except:
        return 0
