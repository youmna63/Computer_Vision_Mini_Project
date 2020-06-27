# Computer_Vision_Mini_Project
import cv2
import numpy as num
from matplotlib import pyplot as pyp
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

image = camera()

edges = cv2.Canny(image, threshold1=80, threshold2=400)
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


x, y = num.ogrid[:200, :200]
image = num.exp(6j * num.hypot(x, y) ** 1.5 / 30.).real


figure, axes = pyp.subplots(nrows=3, ncols=3, sharex=True, sharey=True,
                         figsize=(10, 10))

edge_roberts=filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_scharr = filters.scharr(image)
edge_prewitt = filters.prewitt(image)

different_scharr_roberts = compare_images(edge_scharr, edge_roberts)
different_scharr_prewitt = compare_images(edge_scharr, edge_prewitt)
different_scharr_sobel = compare_images(edge_scharr, edge_sobel)
max_diff = num.max(num.maximum(different_scharr_roberts,different_scharr_prewitt, different_scharr_sobel))


axes = axes.ravel()


axes[0].imshow(image, cmap=pyp.cm.gray)
axes[0].set_title('Original image')

axes[1].imshow(edge_scharr, cmap=pyp.cm.gray)
axes[1].set_title('Scharr Edge Detection')

axes[2].imshow(edge_roberts, cmap=pyp.cm.gray)
axes[2].set_title('Scharr Roberts')


axes[3].imshow(different_scharr_prewitt, cmap=pyp.cm.gray, vmax=max_diff)
axes[3].set_title('Scharr - Prewitt')

axes[4].imshow(different_scharr_sobel, cmap=pyp.cm.gray, vmax=max_diff)
axes[4].set_title('Scharr - Sobel')

for ax in axes:
    ax.axis('off')

pyp.tight_layout()
pyp.show()
