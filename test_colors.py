import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

im = '/Users/Mariana/mieec/Tese/Development/Final_imageRepresentation/color_cube.png'
img = cv2.imread(im,1)
img_gray = cv2.imread(im,0)
img_lab = cv2.cvtColor(img,cv.CV_BGR2Lab)

div = 2

height, width, comp = img_lab.shape
h_region = height/div
w_region = width/div

#for roow in img:
    #for pixel in roow:
        #color = pixel[0]*256*256 + pixel[1]*256 + pixel[2]
        #hist[int(color)] = hist[int(color)] + 1
      
max_colors = []

for i in range(0,div):
    for j in range(0,div):
        
        print 'i='+str(i)
        print 'j='+str(j)
           
	#mask
	mask = np.zeros(img_gray.shape, dtype=np.uint8)
	mask[i*h_region:(i+1)*h_region, j*w_region:(j+1)*w_region] = 1	       

	hist = cv2.calcHist([img_lab],[0,1,2],mask,[256,256,256],[0,256,0,256,0,256])
        
        max_color_l, max_color_a, max_color_b = np.where(hist == np.max(hist))
        max_colors.append([max_color_l[0], max_color_a[0], max_color_b[0]])

#plt.plot(hist[0])
#plt.ylabel('Histogram')
#plt.show()