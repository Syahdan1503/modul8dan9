import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from skimage import data


img = data.camera()

img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
img_sobel =img_sobelx + img_sobely

fig, axes = plt.subplots(4, 2, figsize=(20,20))
ax = axes.ravel()
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Citra Input')
ax[1].hist(img.ravel(), bins =256)
ax[1].set_title("histogram Citra Input")

ax[2].imshow(img_sobelx, cmap='gray')
ax[2].set_title('Citra Output')
ax[3].hist(img_sobelx.ravel(), bins = 256)
ax[3].set_title("histogram Citra Output")

ax[4].imshow(img_sobely, cmap='gray')
ax[4].set_title('Citra Output')
ax[5].hist(img_sobely.ravel(), bins = 256)
ax[5].set_title("histogram Citra Output")

ax[6].imshow(img_sobel, cmap='gray')
ax[6].set_title('Citra Output')
ax[7].hist(img_sobel.ravel(), bins = 256)
ax[7].set_title("histogram Citra Output")

fig.tight_layout()
plt.show()

#Latihan 2filter prewit
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from skimage import data

img = data.camera()

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv2.filter2D(img, -1, kernelx)
img_prewitty = cv2.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

fig, axes = plt.subplots(4, 2, figsize=(20,20))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Citra Input')
ax[1].hist(img.ravel(), bins =256)
ax[1].set_title("histogram Citra Input")

ax[2].imshow(img_prewittx, cmap='gray')
ax[2].set_title('Citra Output Prewittx')
ax[3].hist(img_prewittx.ravel(), bins = 256)
ax[3].set_title("histogram Citra Output Prewittx")

ax[4].imshow(img_prewitty, cmap='gray')
ax[4].set_title('Citra Output Prewitty')
ax[5].hist(img_prewitty.ravel(), bins = 256)
ax[5].set_title("histogram Citra Output Prewitty")

ax[6].imshow(img_prewitt, cmap='gray')
ax[6].set_title('Citra Output Prewitt')
ax[7].hist(img_prewitt.ravel(), bins = 256)
ax[7].set_title("histogram Citra Output Prewitt")

fig.tight_layout()
plt.show()
#latihan 3 filter canny

import cv2 
import numpy as np 
from matplotlib import pyplot as plt
from skimage import data

img = data.camera()

img_canny = cv2.Canny(img, 100, 200)

fig, axes = plt.subplots(2, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Citra Input')
ax[1].hist(img.ravel(), bins =256)
ax[1].set_title("histogram Citra Input")

ax[2].imshow(img_canny, cmap='gray')
ax[2].set_title('Citra Output')
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("histogram Citra Output")
plt.show()

#filter turunan kedua 
import cv2
from matplotlib import pyplot as plt

#baca gambar 
img0 = cv2.imread(r'C:\Users\ASUS\Documents\FILE SYAHDAN\semester 6\PCD\pertemuan 11\mirage.jpg')
#konversi ke gray 
gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
#hilangkan noise
img = cv2.GaussianBlur(gray, (3, 3,), 0)
# Konvolusi dengan kernel
laplacian = cv2.Laplacian(img, cv2.CV_64F)

#tampilkan dengan matplotlib
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(laplacian, cmap ='gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])
plt.show()






