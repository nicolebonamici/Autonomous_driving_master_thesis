# Import the required libraries
import matplotlib.pylab as plt
import cv2
import numpy as np
import os
import shutil

lane_image = cv2.imread('media/image_lane.jpeg')

# Convert the image in to RBG format using matplotlib
colored_image = cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB)

# Find out the shape of the image, defining height (1442) and width (2560)
height = lane_image.shape[0]
width = lane_image.shape[1]

#Define the region of interest
def CannyEdge(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  cannyImage = cv2.Canny(blur, 10, 30)
  return cannyImage


# Create a function called region_of_interest which takes one argument called image
def region_of_interest(image):
  height = image.shape[0]
  triangle = np.array([[(160, height),(600, 370),(1050, height),]], np.int32)
  mask = np.zeros_like(image)
  cv2.fillPoly(mask, triangle, 255)
  masked_image = cv2.bitwise_and(image, mask)
  return masked_image

# Loop over all lines and draw them on the blank image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# Run the code on the video   
cap = cv2.VideoCapture("./media/test.mp4")

# Create out directory for the frame of the resulting video
if not os.path.exists('out'):
    os.makedirs('out')
else:
  shutil.rmtree('out') # delete previous content of out folder
  os.makedirs('out')

frame_number = 0
while(cap.isOpened()):
    frame_is_valid, frame = cap.read()
    if frame_is_valid is False: # stop execution if frame is invalid, typically end of file
      break
    canny = CannyEdge(frame)
    cropped_Image = region_of_interest(canny)
    rho = 2
    theta = np.pi/180
    threshold = 100
    lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([ ]))
    line_image = display_lines(frame, lines)

    combo_image = cv2.addWeighted(frame, 0.6, line_image, 1, 1)
    cv2.imshow("Image", combo_image)
    cv2.imwrite('./out/frame_' + str(frame_number).zfill(7) + '.jpg', combo_image) # write frame to out folder
    frame_number += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()

# save images with ffmpeg command
os.system("ffmpeg -framerate 24 -i ./out/frame_%07d.jpg output.mp4")
