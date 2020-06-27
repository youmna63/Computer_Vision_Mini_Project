# Computer_Vision_Mini_Project
import cv2
import numpy as num
from matplotlib import pyplot as pyp
from matplotlib.style import use
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

x, y = num.ogrid[:200, :200]
image = num.exp(6j * num.hypot(x, y) ** 1.5 / 30.).real


figure, axes = pyp.subplots(nrows=2, ncols=3, sharex=True, sharey=True,
                         figsize=(10, 10))
images = cv2.imread('images.jpg')
gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=80, threshold2=400)
blurred = cv2.GaussianBlur(images,(5,5),3)
laplacian = cv2.Laplacian(images,2)
sobelx = cv2.Sobel(images,cv2.CV_64FC1,0,1)
sobely = cv2.Sobel(images,cv2.CV_64FC1,1,0)

axes = axes.ravel()


axes[0].imshow(images, cmap=pyp.cm.gray)
axes[0].set_title('Original image')

axes[1].imshow(edges, cmap=pyp.cm.gray)
axes[1].set_title('Edge Detection')

axes[2].imshow(blurred, cmap=pyp.cm.gray)
axes[2].set_title('Blurred image')

axes[3].imshow(laplacian, cmap=pyp.cm.gray)
axes[3].set_title('Laplacian')

axes[4].imshow(sobelx, cmap=pyp.cm.gray)
axes[4].set_title('SobelX')

axes[5].imshow(sobely, cmap=pyp.cm.gray)
axes[5].set_title('SobelY')





#pyp.subplot(2,3,2),pyp.imshow(edges,cmap = 'gray')
#pyp.title('Edge Detect'),pyp.xticks([0]), pyp.yticks([2])

#pyp.subplot(2,3,3),pyp.imshow(blurred)
#pyp.title('Blurred'),pyp.xticks([0]), pyp.yticks([3])

#pyp.subplot(2,3,4),pyp.imshow(laplacian,cmap = 'gray')
#pyp.title('Laplacian'), pyp.xticks([0]), pyp.yticks([4])

#pyp.subplot(2,3,5),pyp.imshow(sobelx,cmap = 'gray')
#pyp.title('Sobel X'), pyp.xticks([0]), pyp.yticks([5])

#pyp.subplot(2,3,6),pyp.imshow(sobely,cmap = 'gray')
#pyp.title('Sobel Y'), pyp.xticks([0]), pyp.yticks([6])


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


