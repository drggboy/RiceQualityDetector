import cv2
import numpy as np
import matplotlib.pyplot as plt

# 背景提取，开运算   https://blog.csdn.net/sgzqc/article/details/121000489
# Step 1: Read image
image_file = r'./img/many.png'
# image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(image_file, cv2.IMREAD_COLOR)
selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  #kernel大小，准备进行开运算
background = cv2.morphologyEx(image, cv2.MORPH_OPEN, selem)    #开运算
foreground = cv2.subtract(image, background)

cv2.imshow('image',foreground)
cv2.waitKey(0)

gray = cv2.cvtColor(foreground,cv2.COLOR_BGR2GRAY)
thresh_otsu, binary_image = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('thresh',thresh_otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()


# thresh_otsu, binary_image = cv2.threshold(foreground.astype(np.uint8), 0, 255,
#                 cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# def processImage(fileName):
#   # Load in the image using the typical imread function using our watch_folder path, and the fileName passed in, then set the final output image to our current image for now
#   image = cv2.imread(watch_folder + ‘/’ + fileName)
#   output = image
#   # Set thresholds. Here, we are using the Hue, Saturation, Value color space model. We will be using these values to decide what values to show in the ranges using a minimum and maximum value.
#   # THESE VALUES CAN BE PLAYED AROUND FOR DIFFERENT COLORS
#   hMin = 29  # Hue minimum
#   sMin = 30  # Saturation minimum
#   vMin = 0   # Value minimum (Also referred to as brightness)
#   hMax = 179 # Hue maximum
#   sMax = 255 # Saturation maximum
#   vMax = 255 # Value maximum
#   # Set the minimum and max HSV values to display in the output image using numpys' array function. We need the numpy array since OpenCVs' inRange function will use those.
#   lower = np.array([hMin, sMin, vMin])
#   upper = np.array([hMax, sMax, vMax])
#   # Create HSV Image and threshold it into the proper range.
#   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Converting color space from BGR to HSV
#   mask = cv2.inRange(hsv, lower, upper) # Create a mask based on the lower and upper range, using the new HSV image
#   # Create the output image, using the mask created above. This will perform the removal of all unneeded colors, but will keep a black background.
#   output = cv2.bitwise_and(image, image, mask=mask)
#   # Add an alpha channel, and update the output image variable
#   *_, alpha = cv2.split(output)
#   dst = cv2.merge((output, alpha))
#   output = dst
#   # Resize the image to 512, 512 (This can be put into a variable for more flexibility), and update the output image variable.
#   dim = (512, 512)
#   output = cv2.resize(output, dim)
#   # Generate a random file name using a mini helper function called randomString to write the image data to, and then save it in the processed_folder path, using the generated filename.
#   file_name = randomString(5) + ‘.png’
#   cv2.imwrite(processed_folder + ‘/’ + file_name, output)

cv2.imshow('image',binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()