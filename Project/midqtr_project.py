"""
DSC 20 Mid-Quarter Project
Name(s): Leni Dai, Hongyu Yu
PID(s):  A16495711, A17155271
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3
max_intensity = 255


# Part 1: RGB Image #
class RGBImage:
    """
    A template for image objects in RGB color spaces.
    """

    def __init__(self, pixels):
        """
        The constructor for an RGBImage instance that creates needed instance 
        variables. Taks the argument pixels that is a 3-dimension list 
        represents the image objects. Set the value of num_rows and num_cols 
        which represent the number of rows and number of columns. 


        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

        
        if type(pixels) != list or len(pixels) < 1:
            raise TypeError()
        len_row = len(pixels[0])
        for row in pixels:
            if type(row) != list or len(row) < 1:
                raise TypeError()
            elif len(row) != len_row:
                raise TypeError()
            for i in row:
                if type(i) != list or len(i) != NUM_CHANNELS:
                    raise TypeError()
                for pixe in i:
                    if type(pixe) != int:
                        raise ValueError()
                    elif pixe < 0 or pixe > max_intensity:
                        raise ValueError()

    def size(self):
        """
        The getter method that returns the size of image in a tuple as (number 
        of rows, number of columns).

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        The getter method that returns a DEEP COPY of the pixels matrix.

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        #copied_pixels = []
        #for row in self.pixels:
        #    for element in row:
        #        [[i for i in element] for j in row]
        return [[[p for p in j]for j in i]for i in self.pixels]



    def copy(self):
        """
        A method that returns a copy of the RGBImage instance. Creates a new 
        RGBImage instance using a deep copy of the pixels matrix and returns 
        the new instance.
        
        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Takes two integer arguments row and col. A getter method that returns 
        the color of the pixel at position (row, col).

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        elif row >= self.num_rows or col >= self.num_cols:
            raise ValueError()
        pixel_lst = self.pixels[row][col]
        pixel_tuple = (pixel_lst[0],pixel_lst[1],pixel_lst[2])
        return pixel_tuple

        

    def set_pixel(self, row, col, new_color):
        """
        A setter method that updates the color of the pixel at position (row, 
        col) to the new_color. If any color intensity in the new_color tuple is
        a negative value, do not change the intensity at the corresponding 
        channel.


        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        if type(row) != int or type(col) != int:
            raise TypeError()
        elif row >= self.num_rows or col >= self.num_cols:
            raise ValueError()
        if type(new_color) != tuple or len(new_color) != 3:
            raise TypeError()
        for i in range(3):
            if type(new_color[i]) != int:
                raise TypeError()
            elif new_color[i] > max_intensity:
                raise ValueError()
            elif new_color[i] < 0:
                self.pixels[row][col][i] = self.pixels[row][col][i]
            else:
                self.pixels[row][col][i] = new_color[i]
        return





# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    In this part will create several image processing methods.
    """

    def __init__(self):
        """
        Initializes an ImageProcessingTemplate instance and instance variables.

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost =0

    def get_cost(self):
        """
        Return the current total cost.

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        return  self.cost

    def negate(self, image):
        """
        Returns the negative of the given image.

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        negate_copy_image=image.copy()
        inverted=[[[255-p for p in j]for j in i]for i in \
        negate_copy_image.get_pixels()]
        return RGBImage(inverted)

    def grayscale(self, image):
        """
        Converts the given image to grayscale.

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        grayscale_copy_image=image.copy()

        average_of_pixels=[[int(sum([p for p in j])/3) for j in i]\
        for i in grayscale_copy_image.get_pixels()]
        update_all_channels =[[[j for p in range(3)]for j in i]for \
        i in average_of_pixels]
        return RGBImage(update_all_channels)

    def rotate_180(self, image):
        """
        Rotates the image 180 degrees. 


        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        rotate_image_copied=image.copy()
        rotate_image=[ i[::-1] for i in rotate_image_copied.get_pixels()[::-1]]
        return RGBImage(rotate_image)

# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    In this part, we will create a monetized version of the template class.
    It will update the current total cost after calling each image processing
    method.
    """

    def __init__(self):
        """
        Initializes an StandardImageProcessing instance and instance variables.

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.free_method_calls=0
        self.rotate_num=0

    def negate(self, image):
        """
        Returns the negative of the given image. 
        Every time this method is called, the total cost increases by $5.

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        if self.free_method_calls>0:
            self.cost=0
            self.free_method_calls-=1
        else:
            self.cost+=5
        negate_copy_image=image.copy()
        return ImageProcessingTemplate.negate(self, negate_copy_image)


    def grayscale(self, image):
        """
        Converts the given image to grayscale.
        Every time this method is called, the total cost increases by $6.
        """
        if self.free_method_calls>0:
            self.cost=0
            self.free_method_calls-=1
        else:
            self.cost+=6
        grayscale_copy_image=image.copy()
        return ImageProcessingTemplate.grayscale(self, grayscale_copy_image)

    def rotate_180(self, image):
        """
        Rotates the image 180 degrees. 
        And it should add +10 to the cost each time it is used.
        when rotate_180 undoes itself the method will be free, 
        and the cost of the previous rotate_180 will be refunded. 


        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        if self.free_method_calls>0:
            self.cost=self.cost
            self.free_method_calls-=1
        elif self.rotate_num==0:
            self.cost+=10
            self.rotate_num+=1
        elif self.rotate_num==1:
            self.cost-=10
            self.rotate_num-=1
        rotate_180_copy=image.copy()
        return ImageProcessingTemplate.rotate_180(self,rotate_180_copy)


    def redeem_coupon(self, amount):
        """
        Makes the next amount image processing method calls free.

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """
        if amount<=0:
            raise ValueError()
        elif type(amount)!=int:
            raise TypeError()
        else:
            self.free_method_calls+=amount

# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        TODO: add description

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        if isinstance(chroma_image,RGBImage)==False or \
        isinstance(background_image,RGBImage)==False:
            raise TypeError()
        elif chroma_image.size()!=background_image.size():
            raise ValueError()
        copy_image=chroma_image.copy()

        for i in range(copy_image.num_rows):
            for j in range(copy_image.num_cols):
                if copy_image.get_pixel(i, j) ==color:
                    copy_image.set_pixel(i,j, \
                        background_image.get_pixel(i,j))
        return copy_image

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        TODO: add description

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        if isinstance(sticker_image,RGBImage)==False or \
        isinstance(background_image,RGBImage)==False:
            raise TypeError()
        elif sticker_image.size()>=background_image.size():
            raise ValueError()

        elif type(x_pos)!=int or type(y_pos)!=int:
            raise TypeError
        elif sticker_image.num_rows+x_pos>=background_image.num_rows or \
        sticker_image.num_cols+y_pos>=background_image.num_cols:
            raise ValueError
        background_copy_image=background_image.copy()

        for i in range(background_copy_image.num_rows):
            if i>=x_pos and i<sticker_image.num_rows+x_pos:
                for j in range(background_copy_image.num_cols):
                    if j>=y_pos and j<sticker_image.num_cols+y_pos:
                        background_copy_image.set_pixel(i,j,\
                            sticker_image.get_pixel(i-x_pos, j-y_pos))
        return background_copy_image

# Part 5: Image KNN Classifier #
class ImageKNNClassifier:
    """
    TODO: add description

    # make random training data (type: List[Tuple[RGBImage, str]])
    >>> train = []

    # create training images with low intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
    ...     for _ in range(20)
    ... )

    # create training images with high intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
    ...     for _ in range(20)
    ... )

    # initialize and fit the classifier
    >>> knn = ImageKNNClassifier(5)
    >>> knn.fit(train)

    # should be "low"
    >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    low

    # can be either "low" or "high" randomly
    >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    This will randomly be either low or high

    # should be "high"
    >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
    high
    """
    def __init__(self, n_neighbors):
        """
        TODO: add description
        """
        self.data = []
        self.n_neighbors = n_neighbors

    def fit(self, data):
        """
        TODO: add description
        """
        if len(data) <= self.n_neighbors:
            raise ValueError()
        elif len(self.data) != 0:
            raise ValueError()

        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        TODO: add description
        """
        if isinstance(image1,RGBImage)==False or \
        isinstance(image2,RGBImage)==False:
            raise TypeError()
        elif len(image1.get_pixels()) != len(image2.get_pixels()):
            raise ValueError()
        flat_image1 = [a for i in image1.get_pixels() for j in i for a in j]
        flat_image2 = [a for i in image2.get_pixels() for j in i for a in j]
        euclidean_dis = (sum([(flat_image1[i] - flat_image2[i])**2 for i \
            in range(len(flat_image1))]))**0.5
        return euclidean_dis

    @staticmethod
    def vote(candidates):
        """
        TODO: add description
        """
        count = 0
        label = candidates[0]
        for i in candidates:
            curr_frequency = candidates.count(i)
            if(curr_frequency> count):
                count = curr_frequency
                label = i
        return label
        

    def predict(self, image):
        """
        TODO: add description
        """
        if len(self.data)==0:
            raise ValueError()
        distance_lst = [(i[1], self.distance(i[0], image)) for i in self.data]
        candid = [i[0] for i in sorted(distance_lst,key = lambda x: \
            x[1])[:self.n_neighbors]]
        return self.vote(candid)





    







# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)

def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()
