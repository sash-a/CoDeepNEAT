from Augmentor.Operations import Operation

# Create the new operation by inheriting from the operation superclass
class CustomOperation(Operation):

    # Can accept as many custom parameters as needed
    def __init__(self, probability, par1, par2):
        # Call superclass's constructor (with probability)
        Operation.__init__(self, probability)
        # Set custom operations's attributes here:
        self.par1 = par1
        self.par2 = par2

    # images use the python imaging library (PIL)
    def perform_operation(self, image):

        # Start of operation
        # ...
        # End of operation

        # Return image so that it can be processes by the pipeline
        return image


# Once you have a new operation, you can add it to the existing pipeline like so:

    # p = Augmentor.Pipeline("imagePath")
    # cust = CustomOperation(0.7, 1, 2)
    # p.add_operation(cust)
    # p.sample(100)

# Using non-PIL image objects:

    # import numpy
    # 'Custom class declaration'
    # ...
    # def perform_operation(image):
    #   image_array = numpy.array(image).astype('uint8')
    #   Perform your custom operations here
    #   image = PIL.Image.fromarray(image_array)
    #   return image
