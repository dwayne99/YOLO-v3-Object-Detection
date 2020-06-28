from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

from utils import * 

from pdb import set_trace

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    '''
    The create_modules function takes a list blocks returned by the parse_cfg function.
    '''

    # variable net_info to store information about the network.
    net_info = blocks[0]

    module_list = nn.ModuleList()

    # We initialise this to 3, as the image has 3 filters corresponding to the RGB channels.
    prev_filters = 3
    output_filters = []

    # Now, the idea is to iterate over the list of blocks, and create a PyTorch module for each block as we go.
    for index, block in enumerate(blocks[1:]):
        # nn.Sequential class is used to sequentially execute a number of nn.Module objects. 
        module = nn.Sequential()

        # ckeck the type of block
        # create a new module for the block
        # append to module_list

        ######## CONVOLUTIONAL LAYER ########
        if(block['type'] == 'convolutional'):
            
            # info about the layer
            activation = block['activation']

            #  check if batch normalization exists
            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(block['filters'])
            padding = int(block['pad'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            # padding = 0 -> valid mode
            # padding = 1 -> same mode
            if padding:
                # same mode
                # pad image as follows
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # putting it all together into the conv layer
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module(f"conv_{index}",conv)

            # add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f"batch_norm_{index}", bn)
            
            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1,inplace=True)
                module.add_module(f"leaky_{index}",activ)

        ########## UPSAMPLE LAYER  #############
        elif( block['type'] == 'upsample'):
            
            # for upsampling layer we could use Bilinear2dUpsampling
            '''
            Upsample increases the height and width of a tensor
            input  : (N x C x H_in x W_in)
            output : (N x C x H_out x W_out)

            where  : H_out = H_in * scale_factor
                     W_out = W_in * scale_factor

            example: 
                    >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
                    >>> input
                        tensor([[[[ 1.,  2.],
                                [ 3.,  4.]]]])
                                
                    >>> m = nn.Upsample(scale_factor=2, mode='bilinear')  # align_corners=False
                    >>> m(input)
                    tensor([[[[ 1.0000,  1.2500,  1.7500,  2.0000],
                            [ 1.5000,  1.7500,  2.2500,  2.5000],
                            [ 2.5000,  2.7500,  3.2500,  3.5000],
                            [ 3.0000,  3.2500,  3.7500,  4.0000]]]])
            '''
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode='nearest')
            module.add_module(f'upsample_{index}',upsample)

        ############# ROUTE LAYER ################
        elif( block['type'] == 'route'):

            #The code for creating the Route Layer deserves a fair bit of explanation.
            # At first, we extract the the value of the layers attribute, cast it into
            # an integer and store it in a list.
            # Then we have a new layer called EmptyLayer which, as the name suggests 
            # is just an empty layer.
            
            block['layers'] = block['layers'].split(',')

            # start of a route
            start = int(block['layers'][0])
            # end if there exists one
            try:
                end = int(block['layers'][1])
            except:   
                end = 0

            # Negative index means jth layer backwards from current layer
            # Positive index means the actual index of the layer in the whole network
            # check for positive annotation and if exists change to negative
            # (make it relative to the present layer)
            if start > 0:
                start = start - index 
            if end > 0:
                end = end - index 

            # Emptylayer does nothing. It just acts as an nn.module
            route = EmptyLayer()
            module.add_module(f"route_{index}",route)

            if end < 0:
                # if end exists (when there are two values for layers)
                # number of filters will be sum of start and end filters
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                # otherwise (when there's only one value)
                # number of filters will be sum of start filters
                filters = output_filters[index + start]

        ############# SHORTCUT LAYER  ###################
        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}',shortcut)

        ############### YOLO LAYER  ####################
        elif block['type'] == 'yolo':
            '''
            The network has 9 different achor sizes(H x W) for 3 anchors in each of 
            the detector layers that have different grid sizes.
            The mask decides which achor sizes has to be selected for a particular 
            detector layer.

            Example: 
            mask = 3,4,5
            anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326

            For this layer with the following mask the achors will have sizes as follows:
            anchor_1 : 30,61 ; anchor_1 : 62,45 ; anchor_1 : 59,119 ;
            '''

            # get the mask and anchors and store it in a list
            mask = block['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = block['anchors'].split(',')
            anchors = [int(a) for a in anchors]

            # group the height and width of anchors
            anchors =  [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            # and select only those anchors that are mentioned in the mask
            anchors = [anchors[i] for i in mask]

            # the DetectionLayer simply takes one argument the anchors 
            detection = DetectionLayer(anchors)
            module.add_module(f'Detection_{index}',detection)

        # append the module to the modules_list
        module_list.append(module)

        # update previous filters to current filters
        prev_filters = filters

        # keep track of filters for each layer by appending to output_filters
        output_filters.append(filters)

    return (net_info, module_list)


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)

        # module_list contains the modules of the network in sequence
        self.net_info, self.module_list = create_modules(self.blocks)
        

    def forward(self,x,CUDA):

        '''
        The forward function takes two arguments:
        Args :  x    -> input
                CUDA -> Flag which signifies whether to use GPU or CPU

        Desc : The function runs the input through the various modules of the network 
               and returns the bounding boxes 

        Returns : Tensor with a shape of 1 x 10647 x 85. The first dimension 
                  is the batch size which is simply 1 because we have used a single 
                  image. For each image in a batch, we have a 10647 x 85 table. 
                  The row of each of this table represents a bounding box. 
                  (4 bbox attributes, 1 objectness score, and 80 class scores)
        '''

        # store info of each module being used
        modules = self.blocks[1:]
        # We cache the outputs for the route layer in dict outputs, since route and 
        # shortcut layers need output maps from previous layers.
        outputs = {}
        
        # write flag is used to indicate if we have or haven't initialized output
        write = 0

        for i, module in enumerate(modules):

            # check the module type that we're currently at in the network
            module_type = (module['type'])

            ################ CONVOLUTIONAL OR UPSAMPLE LAYER ###################
            if module_type == 'convolutional' or module_type == 'upsample':
                # pass input or output from previous layer through conv or upsample layer
                x = self.module_list[i](x)

            ######################### ROUTE LAYER ##########################
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]

                # convert layer to relative indexing with present layer
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i 

                # obtain features from layer relative to current layer
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                # if there are two layers mentioned
                else:
                    # conver layer to relative index
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    #In PyTorch, input and output of a convolutional layer has the format 
                    # `B X C X H X W. The depth corresponding the the channel dimension 
                    # which is dim=1
                    x = torch.cat((map1,map2),1)

            elif module_type == 'shortcut':
                # it is the residue of resnet
                # adds output from a particular layer(from_) to the previous layer
                from_ = int(module['from'])
            

                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == 'yolo':
                # this is the detection layer

                # 0 index in module list is the detection layer in Sequenctial module
                anchors = self.module_list[i][0].anchors  # ex: [(116, 90), (156, 198), (373, 326)]


                # get the input dimentions(height==weight==416)
                inp_dim = int(self.net_info['height'])

                # get the number of classes(==80)
                num_classes = int(module['classes'])

                # transform 
                x = x.data
                
                # print(f'Size before transform =>{x.size()}')
                # convert output to 2D
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # print(f'Size after transform =>{x.size()}')

                if not write: # i.e if no collector has been initialized
                    detections = x 
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections


    def load_weights(self, weightfile):
        '''
        input : weightfile path

        note: The weights are just stored as floats, with nothing to guide us as
              to which layer do they belong to. Since, you're reading only floats, 
              there's no way to discriminate between which weight belongs to 
              which layer. Hence, we must understand how the weights are stored.

        '''

        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)




'''
def get_test_input():
    img = cv2.imread("filename.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
'''
      
'''
### TEST parse_cfg function
blocks = parse_cfg('cfg/yolov3.cfg')
print(create_modules(blocks))
'''

'''
### TEST create_modules function
model = Darknet('cfg/yolov3.cfg')
print(model)
'''


'''
### TEST Model
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable parameters = {param_count}')
image = cv2.imread('filename.png')
inp = get_test_input(image,416)
if torch.cuda.is_available():
    model = model.cuda()
    inp = inp.cuda()
with torch.no_grad():
    pred = model(inp, torch.cuda.is_available())
print (pred)
set_trace()
prediction = write_results(pred, 0.5, 80, nms_conf = 0.4)
set_trace()
'''