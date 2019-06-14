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
        inputs = []
        inputType = None
        numFeatures = -1

        inputShapes = ""
        for parent in self.moduleNodeInputIDs:
            input = self.accountedForInputIDs[parent.traversalID]


            #combine inputs

            if(inputType is None):
                inputType = type(parent.deepLayer)
                numFeatures = self.getOutFeatures(deepLayer=parent.deepLayer), input.size()[2], input.size()[3]
            else:
                if(type(parent.deepLayer) == inputType):
                    #continue merging
                    newNumFeatures = self.getOutFeatures(deepLayer=parent.deepLayer), input.size()[2], input.size()[3]
                    #print("merging two layers of the same type feature counts:", newNumFeatures, numFeatures, "inputs:",inputShapes)

                    if(newNumFeatures == numFeatures):
                        #can sum
                        pass
                    else:
                        #different input shapes
                        #print("trying to merge two",inputType, "feature sizes of:",newNumFeatures,"and",numFeatures)
                        if(inputType == nn.Conv2d):
                            #print("merging conv layers")
                            channels1, x1,y1 = numFeatures
                            channels2, x2,y2 = newNumFeatures
                            if(not channels1 == channels2):
                                print("trying to merge two conv layers with differing numbers of channels :",channels1, channels2)
                                return
                            else:
                                if(x1 < x2):
                                    #print("new image thinner than previous on x")
                                    #previous inputs are smalller on the x axis
                                    leftPad = (x2 - x1) // 2
                                    rightPad = (x2 - x1) - leftPad
                                    for i in range(len(inputs)):
                                        #print("changing previous from",inputs[i].size(), end=" ")
                                        inputs[i] = F.pad(input=inputs[i], pad = (0,0,leftPad, rightPad), mode='constant', value=0)
                                        #print("to",inputs[i].size())

                                elif(x2<  x1):
                                    #print("new image wider than previous on x")
                                    #new found input is smaller on x than previous
                                    leftPad = (x1 - x2)//2
                                    rightPad = (x1 - x2) - leftPad

                                    input = F.pad(input=input, pad = (0,0,leftPad, rightPad), mode='constant', value=0)

                                if (y1 < y2):
                                    #print("new image thinner than previous on y")
                                    # previous inputs are smalller on the x axis
                                    leftPad = (y2 - y1) // 2
                                    rightPad = (y2 - y1) - leftPad
                                    for i in range(len(inputs)):
                                        #print("changing previous from", inputs[i].size(), end=" ")
                                        inputs[i] = F.pad(input=inputs[i], pad=(leftPad, rightPad),
                                                          mode='constant', value=0)
                                        #print("to", inputs[i].size())

                                elif (y2 < y1):
                                    #print("new image wider than previous on y")
                                    # new found input is smaller on x than previous
                                    leftPad = (y1 - y2) // 2
                                    rightPad = (y1 - y2) - leftPad

                                    input = F.pad(input=input, pad=( leftPad, rightPad), mode='constant', value=0)


                        elif(inputType == nn.Linear):
                            print("merging linear layers with different layer counts")
                        else:
                            print("not yet implemented merge of layer type:",inputType)
                else:
                    print("trying to merge layers of different types:",type(parent.deepLayer),";", inputType,"this has not been implemented yet")
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


