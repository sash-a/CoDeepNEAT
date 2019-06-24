import torch.nn.functional as F
import torch


def merge_linear_outputs( previous_inputs, new_input, cat = False):
    print("merging linear layers with different feature counts")
    if(cat):
        previous = torch.sum(torch.stack(previous_inputs), dim=0)
        return [torch.cat([previous, new_input],dim=0)]
    else:

        new_input, previous_inputs= pad_linear_outputs(previous_inputs, new_input)
        previous_inputs.append(new_input)
        return previous_inputs

    
def pad_linear_outputs(previous_inputs, new_input):
    sizeDiff = previous_inputs[0].out_features - new_input.out_features
    if(sizeDiff > 0):
        #previous is larger
        for i in range(len(previous_inputs)):
            previous_inputs[i] = F.pad(input=previous_inputs[i], pad=sizeDiff)
    else:
        #new is larger
        new_input = F.pad(input=new_input, pad= -sizeDiff)

    return new_input, previous_inputs

def merge_conv_outputs(previous_num_features, previous_inputs, new_num_features, new_input):
    #print("merging conv tensors",len(previous_inputs), previous_inputs[0].size(),new_input.size())
    # conv layers here do not have
    channels1, x1, y1 = previous_num_features
    channels2, x2, y2 = new_num_features
    if channels1 != channels2:
        print("trying to merge two conv layers with differing numbers of channels :", channels1, channels2)
        return
    else:

        size_ratio = (x1 + y1) / (x2 + y2)
        if size_ratio < 1:
            size_ratio = 1 / size_ratio

        if round(size_ratio) > 1.45:#a ratio less than 1.45 will be made worse by maxPooling, requiring even more padding
            # tensors are significantly different - should use a maxPool here to shrink the larger of the two
            #print("using max pooling for prev:", x1,y1,"new:",x2,y2)
            new_input, previous_inputs = max_pool_conv_input(x1, x2, y1, y2, new_input, previous_inputs)
            x1,y1, x2, y2 = previous_inputs[0].size()[2],  previous_inputs[0].size()[3],new_input.size()[2], new_input.size()[3]
            if x1 != x2 or y1 != y2:
                #print("using padding for prev:", x1,y1,"new:",x2,y2)
                #larger convs have been pooled. however a small misalignment remains
                new_input, previous_inputs = pad_conv_input(x1, x2, y1, y2, new_input, previous_inputs)
            x1,y1, x2, y2 = new_input.size()[2], new_input.size()[3], previous_inputs[0].size()[2],  previous_inputs[0].size()[3]
            #print("returning prev:", x1, y1, "new:", x2, y2)


        else:
            # tensors are similar size - can be padded
            #print("using padding, prev:", x1,y1,"new:",x2,y2)
            new_input, previous_inputs = pad_conv_input(x1, x2, y1, y2, new_input, previous_inputs)

    previous_inputs.append(new_input)
    return previous_inputs


def max_pool_conv_input(x1, x2, y1, y2, new_input, previous_inputs):
    """takes a new input, and a list of homogenous previousInputs"""
    size_ratio = (x1 + y1) / (x2 + y2)
    if size_ratio < 1:
        size_ratio = 1 / size_ratio

    if (x1 + y1) > (x2 + y2):
        # previous inputs must be pooled
        #print("pooling prev")
        for i in range(len(previous_inputs)):
            previous_inputs[i] = F.max_pool2d(previous_inputs[i], kernel_size=(round(size_ratio), round(size_ratio)))

    else:
        #print("pooling new")
        new_input = F.max_pool2d(new_input, kernel_size=(round(size_ratio), round(size_ratio)))

    return new_input, previous_inputs

def pad_conv_input(x1, x2, y1, y2, new_input, previous_inputs):
    if x1 < x2:
        # previous inputs are smalller on the x axis
        left_pad = (x2 - x1) // 2
        right_pad = (x2 - x1) - left_pad
        for i in range(len(previous_inputs)):
            previous_inputs[i] = F.pad(input=previous_inputs[i], pad=(0, 0, left_pad, right_pad), mode='constant', value=0)

    elif x2 < x1:
        # new found input is smaller on x than previous
        left_pad = (x1 - x2) // 2
        right_pad = (x1 - x2) - left_pad

        new_input = F.pad(input=new_input, pad=(0, 0, left_pad, right_pad), mode='constant', value=0)

    if y1 < y2:
        # previous inputs are smalller on the x axis
        left_pad = (y2 - y1) // 2
        right_pad = (y2 - y1) - left_pad
        for i in range(len(previous_inputs)):
            previous_inputs[i] = F.pad(input=previous_inputs[i], pad=(left_pad, right_pad),
                                       mode='constant', value=0)

    elif y2 < y1:
        # new found input is smaller on x than previous
        left_pad = (y1 - y2) // 2
        right_pad = (y1 - y2) - left_pad

        new_input = F.pad(input=new_input, pad=(left_pad, right_pad), mode='constant', value=0)

    return new_input, previous_inputs