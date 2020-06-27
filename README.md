# Computer_Vision_Mini_Project
import cv2
import numpy as num
from matplotlib import pyplot as pyp
from matplotlib.style import use
from skimage import filters
from skimage.data import camera
from skimage.util import compare_images

images=camera()
x, y = num.ogrid[:200, :200]
image = num.exp(6j * num.hypot(x, y) ** 1.5 / 30.).real


figure, axes = pyp.subplots(nrows=5, ncols=4, sharex=True, sharey=True,
                         figsize=(15, 15))

edges = cv2.Canny(images, 80, 400)
blurred = cv2.GaussianBlur(images,(5,5),50)
laplacian = cv2.Laplacian(images,2)
sobelx = cv2.Sobel(images,cv2.CV_64FC1,0,1)
sobely = cv2.Sobel(images,cv2.CV_64FC1,1,0)
edge_roberts=filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_scharr = filters.scharr(image)
edge_prewitt = filters.prewitt(image)

different_scharr_roberts = compare_images(edge_scharr, edge_roberts)
different_scharr_prewitt = compare_images(edge_scharr, edge_prewitt)
different_scharr_sobel = compare_images(edge_scharr, edge_sobel)
max_diff = num.max(num.maximum(different_scharr_roberts,different_scharr_prewitt, different_scharr_sobel))

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

axes[7].imshow(image, cmap=pyp.cm.gray)
axes[7].set_title('Original image')

axes[8].imshow(edge_scharr, cmap=pyp.cm.gray)
axes[8].set_title('Scharr Edge Detection')

axes[9].imshow(edge_roberts, cmap=pyp.cm.gray)
axes[9].set_title('Scharr Roberts')

axes[10].imshow(different_scharr_prewitt, cmap=pyp.cm.gray, vmax=max_diff)
axes[10].set_title('Scharr - Prewitt')

axes[11].imshow(different_scharr_sobel, cmap=pyp.cm.gray, vmax=max_diff)
axes[11].set_title('Scharr - Sobel')

for ax in axes:
    ax.axis('off')

pyp.tight_layout()
pyp.show()
