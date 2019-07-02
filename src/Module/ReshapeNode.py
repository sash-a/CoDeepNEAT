
class ReshapeNode:

    def __init__(self, input_shape, output_shape):
        print("creating new reshape node with in:",input_shape, " out:", output_shape)
        self.output_shape = output_shape

    def shape(self, input):

        #print("reshaping input:", input.size())
        batch_size = list(input.size())[0]
        if not batch_size == self.output_shape[0]:
            self.output_shape[0] = batch_size

        if (len(self.output_shape) == 2):
            #print("reshaping input:", input.size(), "to", input.view(self.output_shape[0],self.output_shape[1]).size())
            return input.view(self.output_shape[0],self.output_shape[1])

        if (len(self.output_shape) == 4):
            return input.view(self.output_shape[0],self.output_shape[1],self.output_shape[2],self.output_shape[3])