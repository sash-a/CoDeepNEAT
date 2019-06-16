from src.Module.ModuleNode import ModuleNode as Module
from src.Learner.Layers import MergeSum
import torch
import torch.nn as nn
import torch.nn.functional as F

class AggregatorNode(Module):

    aggregationType = ""
    moduleNodeInputIDs = []# a list of the ID's of all the modules which pass their output to this aggregator
    accountedForInputIDs = {}#map from ID's to input vectors

    def __init__(self):
        Module.__init__(self)
        self.moduleNodeInputIDs = []
        self.accountedForInputIDs = {}
        #print("created aggregator node")

    def insertAggregatorNodes(self, state="start"):
        #if aggregator node has already been created - then the multi parent situation has already been dealt with here
        #since at the time the decendants of this aggregator node were travered further already, there is no need to traverse its decendants again
        pass

    def getParameters(self, parametersDict):
        for child in self.children:
            child.getParameters(parametersDict)

    def addParent(self, parent):
        self.moduleNodeInputIDs.append(parent)
        self.parents.append(parent)

    def resetNode(self):
        """typically called between uses of the forward method of the NN created
            tells the aggregator a new pass is underway and all inputs must again be waited for to pass forward
        """
        self.accountedForInputIDs = {}

    def passANNInputUpGraph(self, input, parentID = ""):
        self.accountedForInputIDs[parentID] = input

        #print("agg(",self.traversalID,") received in:",input, "from",parentID)

        if(len(self.moduleNodeInputIDs) == len(self.accountedForInputIDs)):
            #all inputs have arrived
            #may now aggregate and pass upwards

            #print("agg(",self.traversalID,") received all inputs",self.accountedForInputIDs)

            out =  super(AggregatorNode, self).passANNInputUpGraph(None)
            # if(not out is None):
            #     print("agg got non null out")
            self.resetNode()

            return out

    def passInputThroughLayer(self, _):
        #print("aggregate inputs not yet implemented fully")
        output = None
        inputs = []#method ensures that inputs is always homogenous as new inputs are added
        inputType = None
        numFeatures = -1

        inputShapes = ""
        for parent in self.moduleNodeInputIDs:
            deepLayer = parent.deepLayer
            input = self.accountedForInputIDs[parent.traversalID]
            #combine inputs

            if(inputType is None):
                #first input. no issue by default
                inputType = type(deepLayer)
                numFeatures = self.getOutFeatures(deepLayer=parent.deepLayer), input.size()[2], input.size()[3]
            else:
                if(type(deepLayer) == inputType):
                    #same layer type as seen up till now
                    newNumFeatures = self.getOutFeatures(deepLayer=deepLayer), input.size()[2], input.size()[3]
                    #print("merging two layers of the same type feature counts:", newNumFeatures, numFeatures, "inputs:",inputShapes)

                    if(newNumFeatures == numFeatures):
                        #no issue can sum
                        #print("easy merge")
                        pass
                    else:
                        # different input shapes
                        # print("trying to merge two",inputType, "feature sizes of:",newNumFeatures,"and",numFeatures)
                        if (inputType == nn.Conv2d):
                            #print("merging conv layers")
                            input, inputs = self.mergeConvLayerOutputs(numFeatures, newNumFeatures, input, inputs)

                        elif(inputType == nn.Linear):
                            print("merging linear layers with different layer counts")
                        else:
                            print("not yet implemented merge of layer type:",inputType)
                else:
                    print("trying to merge layers of different types:",type(deepLayer),";", inputType,"this has not been implemented yet")
            #print(type(parent.deepLayer) == nn.Conv2d)
            inputs.append(input)
            inputShapes += "," + repr(input.size())

        #print("in:", inputs)
        #print("stack:", torch.stack(inputs))
        #print("summing:",inputShapes)
        output = torch.sum(torch.stack(inputs), dim=0)
        #print("out:", output)

        return output

    def getPlotColour(self):
        #print("plotting agg node")
        return 'bo'

    def mergeConvLayerOutputs(self, numFeatures, newNumFeatures, input, inputs):
        #print("merging two diff conv tensors")
        #conv layers here do not have
        channels1, x1, y1 = numFeatures
        channels2, x2, y2 = newNumFeatures
        if (not channels1 == channels2):
            print("trying to merge two conv layers with differing numbers of channels :", channels1,channels2)
            return
        else:
            sizeRatio = (x1+y1)/ (x2+y2)
            if(sizeRatio<1):
                sizeRatio = 1/sizeRatio

            #print("size ratio:",sizeRatio, round(sizeRatio))

            if(round(sizeRatio) > 1):
                # tensors are significantly different - should use a maxPool here to shrink the larger of the two
                #print("pooling to merge conv layers")
                if ((x1 + y1) > (x2 + y2)):
                    # previous inputs must be pooled
                    for i in range(len(inputs)):
                        inputs[i] = F.max_pool2d(inputs[i], kernel_size=(round(sizeRatio), round(sizeRatio)))
                        numFeatures = channels1, inputs[i].size()[2], inputs[i].size()[3]

                else:
                    input = F.max_pool2d(input, kernel_size=(round(sizeRatio), round(sizeRatio)))
                    newNumFeatures = channels2, input.size()[2], input.size()[3]

                channels1, x1, y1 = numFeatures
                channels2, x2, y2 = newNumFeatures
                if(not x1 ==x2 or not y1 == y2):
                    #print("padding after pooling to merge conv layers")
                    input, inputs = self.padConvInput(x1, x2, y1, y2, input, inputs)

            else:
                # tensors are similar size - can be padded
                #print("padding to merge conv layers")
                input, inputs = self.padConvInput(x1, x2, y1, y2, input, inputs)

        return input, inputs

    def padConvInput(self,x1,x2,y1,y2, newInput, inputs):
        if (x1 < x2):
            # print("new image thinner than previous on x")
            # previous inputs are smalller on the x axis
            leftPad = (x2 - x1) // 2
            rightPad = (x2 - x1) - leftPad
            for i in range(len(inputs)):
                # print("changing previous from",inputs[i].size(), end=" ")
                inputs[i] = F.pad(input=inputs[i], pad=(0, 0, leftPad, rightPad), mode='constant', value=0)
                # print("to",inputs[i].size())

        elif (x2 < x1):
            # print("new image wider than previous on x")
            # new found input is smaller on x than previous
            leftPad = (x1 - x2) // 2
            rightPad = (x1 - x2) - leftPad

            newInput = F.pad(input=newInput, pad=(0, 0, leftPad, rightPad), mode='constant', value=0)

        if (y1 < y2):
            # print("new image thinner than previous on y")
            # previous inputs are smalller on the x axis
            leftPad = (y2 - y1) // 2
            rightPad = (y2 - y1) - leftPad
            for i in range(len(inputs)):
                # print("changing previous from", inputs[i].size(), end=" ")
                inputs[i] = F.pad(input=inputs[i], pad=(leftPad, rightPad),
                                  mode='constant', value=0)
                # print("to", inputs[i].size())

        elif (y2 < y1):
            # print("new image wider than previous on y")
            # new found input is smaller on x than previous
            leftPad = (y1 - y2) // 2
            rightPad = (y1 - y2) - leftPad

            newInput = F.pad(input=newInput, pad=(leftPad, rightPad), mode='constant', value=0)

        return newInput,inputs

