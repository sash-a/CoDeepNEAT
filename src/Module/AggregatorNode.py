from src.Module.ModuleNode import ModuleNode as Module
import torch
import torch.nn as nn
import torch.nn.functional as F


class AggregatorNode(Module):
    aggregationType = ""
    module_node_input_ids = []  # a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}  # map from ID's to input vectors

    def __init__(self):
        Module.__init__(self)
        self.module_node_input_ids = []
        self.accountedForInputIDs = {}

    def insert_aggregator_nodes(self, state="start"):
        # if aggregator node has already been created - then the multi parent situation has already been dealt with here
        # since at the time the decendants of this aggregator node were travered further already, there is no need to
        # traverse its decendants again
        pass

    def get_parameters(self, parameters_dict):
        for child in self.children:
            child.get_parameters(parameters_dict)

    def add_parent(self, parent):
        self.module_node_input_ids.append(parent)
        self.parents.append(parent)

    def reset_node(self):
        """typically called between uses of the forward method of the NN created
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def pass_ann_input_up_graph(self, input, parent_id=""):
        self.accountedForInputIDs[parent_id] = input

        if len(self.module_node_input_ids) == len(self.accountedForInputIDs):
            # all inputs have arrived
            # may now aggregate and pass upwards
            out = super(AggregatorNode, self).pass_ann_input_up_graph(None)
            # if(not out is None):
            #     print("agg got non null out")
            self.reset_node()

            return out

    def pass_input_through_layer(self, _):
        output = None
        inputs = []  # method ensures that inputs is always homogenous as new inputs are added
        input_type = None
        num_features = -1

        input_shapes = ""
        for parent in self.module_node_input_ids:
            deep_layer = parent.deepLayer
            input = self.accountedForInputIDs[parent.traversalID]
            # combine inputs

            if input_type is None:
                # first input. no issue by default
                input_type = type(deep_layer)
                num_features = self.get_out_features(deep_layer=parent.deepLayer), input.size()[2], input.size()[3]
            else:
                if type(deep_layer) == input_type:
                    # same layer type as seen up till now
                    new_num_features = self.get_out_features(deep_layer=deep_layer), input.size()[2], input.size()[3]

                    if new_num_features == num_features:
                        # no issue can sum
                        pass
                    else:
                        # different input shapes
                        if input_type == nn.Conv2d:
                            # print("merging conv layers")
                            input, inputs = self.merge_conv_outputs(num_features, new_num_features, input, inputs)

                        elif input_type == nn.Linear:
                            print("merging linear layers with different layer counts")
                        else:
                            print("not yet implemented merge of layer type:", input_type)
                else:
                    print("trying to merge layers of different types:", type(deep_layer), ";", input_type,
                          "this has not been implemented yet")
            inputs.append(input)
            input_shapes += "," + repr(input.size())

        output = torch.sum(torch.stack(inputs), dim=0)

        return output

    def get_plot_colour(self):
        return 'bo'

    # TODO only do this check on the first pass through
    def merge_conv_outputs(self, num_features, new_num_features, input, inputs):
        # print("merging two diff conv tensors")
        # conv layers here do not have
        channels1, x1, y1 = num_features
        channels2, x2, y2 = new_num_features
        if channels1 != channels2:
            print("trying to merge two conv layers with differing numbers of channels :", channels1, channels2)
            return
        else:
            size_ratio = (x1 + y1) / (x2 + y2)
            if size_ratio < 1:
                size_ratio = 1 / size_ratio

            if round(size_ratio) > 1:
                # tensors are significantly different - should use a maxPool here to shrink the larger of the two
                if (x1 + y1) > (x2 + y2):
                    # previous inputs must be pooled
                    for i in range(len(inputs)):
                        inputs[i] = F.max_pool2d(inputs[i], kernel_size=(round(size_ratio), round(size_ratio)))
                        num_features = channels1, inputs[i].size()[2], inputs[i].size()[3]

                else:
                    input = F.max_pool2d(input, kernel_size=(round(size_ratio), round(size_ratio)))
                    new_num_features = channels2, input.size()[2], input.size()[3]

                channels1, x1, y1 = num_features
                channels2, x2, y2 = new_num_features
                if x1 != x2 or y1 != y2:
                    input, inputs = self.pad_conv_input(x1, x2, y1, y2, input, inputs)

            else:
                # tensors are similar size - can be padded
                input, inputs = self.pad_conv_input(x1, x2, y1, y2, input, inputs)

        return input, inputs

    def pad_conv_input(self, x1, x2, y1, y2, new_input, inputs):
        if x1 < x2:
            # previous inputs are smalller on the x axis
            left_pad = (x2 - x1) // 2
            right_pad = (x2 - x1) - left_pad
            for i in range(len(inputs)):
                inputs[i] = F.pad(input=inputs[i], pad=(0, 0, left_pad, right_pad), mode='constant', value=0)

        elif x2 < x1:
            # new found input is smaller on x than previous
            left_pad = (x1 - x2) // 2
            right_pad = (x1 - x2) - left_pad

            new_input = F.pad(input=new_input, pad=(0, 0, left_pad, right_pad), mode='constant', value=0)

        if y1 < y2:
            # previous inputs are smalller on the x axis
            left_pad = (y2 - y1) // 2
            right_pad = (y2 - y1) - left_pad
            for i in range(len(inputs)):
                inputs[i] = F.pad(input=inputs[i], pad=(left_pad, right_pad),
                                  mode='constant', value=0)

        elif y2 < y1:
            # new found input is smaller on x than previous
            left_pad = (y1 - y2) // 2
            right_pad = (y1 - y2) - left_pad

            new_input = F.pad(input=new_input, pad=(left_pad, right_pad), mode='constant', value=0)

        return new_input, inputs
