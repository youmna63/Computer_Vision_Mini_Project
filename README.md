# Computer_Vision_Mini_Project
import cv2
import numpy as np
from matplotlib import pyplot as pyp

image = cv2.imread('images.jpg',1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=80, threshold2=400)

pyp.subplot(131),pyp.imshow(image,cmap = 'gray')
pyp.title('Original Image'), pyp.xticks([]), pyp.yticks([])
pyp.subplot(132),pyp.imshow(edges,cmap = 'gray')
pyp.title('Edge Image')
pyp.xticks([]), pyp.yticks([])

pyp.show()


