# Computer_Vision_Mini_Project
import cv2
import numpy as np
from matplotlib import pyplot as pyp

image = cv2.imread('images.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=80, threshold2=400)
blurred = cv2.GaussianBlur(image,(5,5),3)
laplacian = cv2.Laplacian(image, cv2.CV_64F)
sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

pyp.subplot(3,3,1),pyp.imshow(image,cmap = 'gray')
pyp.title('Original'), pyp.xticks([]), pyp.yticks([])

pyp.subplot(3,3,2),pyp.imshow(edges,cmap = 'gray')
pyp.title('Edge Detect'),pyp.xticks([]), pyp.yticks([])

pyp.subplot(3,3,3),pyp.imshow(blurred)
pyp.title('Blurred'),pyp.xticks([]), pyp.yticks([])

pyp.subplot(3,3,4),pyp.imshow(laplacian,cmap = 'gray')
pyp.title('Laplacian'), pyp.xticks([]), pyp.yticks([])

pyp.subplot(3,3,5),pyp.imshow(sobelx,cmap = 'gray')
pyp.title('Sobel X'), pyp.xticks([]), pyp.yticks([])

pyp.subplot(3,3,6),pyp.imshow(sobely,cmap = 'gray')
pyp.title('Sobel Y'), pyp.xticks([]), pyp.yticks([])

pyp.show()



