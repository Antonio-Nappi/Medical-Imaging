import copy

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import collections
import math
import sys
def nothing(x):
    pass
def compute_Laplacian(img):
    lap = cv.Laplacian(img, cv.CV_64F)
    abs = cv.convertScaleAbs(lap)
    return abs
def gammaCorrection(img, gamma):
    table = np.array([(255**(1-gamma))*(i**gamma)
                          for i in range(256)]).astype("uint8")
    return cv.LUT(img, table)
#img = cv.imread("./mdb007.pgm")
#img = cv.resize(img,(100,100))
#214
#np.set_printoptions(threshold=sys.maxsize)
#ch_color = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#mass = np.uint8([180,180,180])
#first = np.where(img <= mass,255,img)
#mass = np.uint8([175,175,175])
#second = np.where(img >= mass,255,img)
#img = first+second
#sample = open("prova.txt","w")
#print(img,file=sample)
#sample.close()
#cv.createTrackbar("lower","Test",0,255,nothing)
#cv.createTrackbar("upper","Test",255,255,nothing)
img = cv.imread("dataset/images/mass/20586934_6c613a14b80a8591_MG_L_CC_ANON.tif",cv.IMREAD_GRAYSCALE)
#img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
str_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
low_bound = np.percentile(img,15)
max = np.amax(img)
min = np.amin(img)
up_bound = np.percentile(img,95)
start = img
#for i in range(img.shape[0]):
#    for j in range(img.shape[1]):
#        str_img[i, j] = (img[i,j])*((up_bound-low_bound)/(max-min))+low_bound
#start = img
histogram = cv.calcHist([img], [0], None, [256], [0, 256])
#plt.plot(histogram)
#plt.show()

start = cv.resize(start,(500,500))
img = cv.resize(img,(500,500))
str_img_r = np.zeros((img.shape[0], img.shape[1]), np.uint8)
out = np.zeros((img.shape[0], img.shape[1]), np.uint8)
abso = np.zeros((img.shape[0], img.shape[1]), np.uint8)
max_r = np.amax(img)
min_r = np.amin(img)
low_bound_r = np.percentile(img,15)
up_bound_r = np.percentile(img,85)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        str_img_r[i, j] = (img[i,j])*((up_bound_r-low_bound_r)/(max_r-min_r))+low_bound_r
max = np.amax(str_img_r)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        out[i,j] = (255/math.log(9*max,10))*math.log(15 + 9*str_img_r[i,j],10)

max = np.amax(out)
min = np.amin(out)

#for i in range(out.shape[0]):
#    for j in range(out.shape[1]):
#        out[i, j] = 255*((out[i,j]-max)/min)
if False:
    print("Dopo di portare gli elementi di nuovo in 0-255")
    unique_elements, counts_elements = np.unique(out, return_counts=True)
    print(unique_elements,counts_elements)
    img3 = cv.GaussianBlur(out, (3, 3), cv.BORDER_DEFAULT)
    img2 = cv.Sobel(out, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    img3 = cv.Sobel(out, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(img2)
    abs_grad_y = cv.convertScaleAbs(img3)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

'''cv.namedWindow('image')
cv.createTrackbar('k', 'image', 1, 100, nothing)
img3 = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
img2 = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
img3 = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(img2)
abs_grad_y = cv.convertScaleAbs(img3)
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
img = img + grad
img3 = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
img2 = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
img3 = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(img2)
abs_grad_y = cv.convertScaleAbs(img3)
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
img = img-grad
out = start-grad
'''


'''print("Value: ",v)
cv.namedWindow("BinoutvsBinOUT")
cv.createTrackbar('k', 'BinoutvsBinOUT', 0, 255, nothing)
img3 = cv.GaussianBlur(out, (3, 3), cv.BORDER_DEFAULT)
img2 = cv.Sobel(out, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
img3 = cv.Sobel(out, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(img2)
abs_grad_y = cv.convertScaleAbs(img3)
grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
out = out + grad'''
'''# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(bin_OUT,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
#cv.watershed()
'''
gamma_out = gammaCorrection(out, 10)
out = 255-out
somma = gamma_out+out
differenza=gamma_out-out
differenza2 = out-gamma_out
v,bin=cv.threshold(differenza,150,1,cv.THRESH_BINARY)
tmp = bin*differenza
v,tmp=cv.threshold(tmp,1,255,cv.THRESH_BINARY)
v,bin_bis=cv.threshold(differenza2,150,255,cv.THRESH_BINARY)
#mean_filter = cv.pyrMeanShiftFiltering(img,10,10)
OUT = (out-str_img_r)
NOTOUT = 255-OUT

while 1:
    #lower = cv.getTrackbarPos("lower","Test")
    #upper = cv.getTrackbarPos("upper","Test")
    #lower = np.uint8([lower,lower,lower])
    #upper = np.uint8([upper,upper,upper])
    #first = np.where(img <= lower, 0, img)
    #second = np.where(img >= upper, 255, img)
    #k=cv.getTrackbarPos('k',"BinoutvsBinOUT")
    cv.imshow("out",out)
    cv.imshow("imageVSStretch", np.hstack((start,str_img_r)))
    cv.imshow("stretchVSLog", np.hstack((str_img_r,out)))
    cv.imshow("imageVSLog", np.hstack((start, OUT)))
    cv.imshow("outvsOUT", np.hstack((out,OUT)))
    #cv.imshow("OPENINGvsBinOUT", np.hstack((opening, bin_OUT)))
    #cv.imshow("SUREFGvsBinOUT", np.hstack((sure_fg, bin_OUT)))
    #cv.imshow("SUREBGvsBINOUT", np.hstack((sure_bg, bin_OUT)))
    #cv.imshow("UNKnown",unknown)
    #cv.imshow("Img",img)
    cv.imshow("Immagini di partenza", np.hstack((out, gamma_out)))
    cv.imshow("Somma", somma)
    cv.imshow("Gamma-out", differenza)
    cv.imshow("out-gamma", differenza2)
    cv.imshow("Gamma-out bin", bin)
    cv.imshow("out-gamma bin", bin_bis)
    #cv.imshow("gradiente",grad)
    #cv.imshow("Mean filter", mean_filter)
    cv.imshow("stretchVSLog", np.hstack((str_img_r,out)))
    cv.imshow("imageVSLog", np.hstack((start, OUT)))
    cv.imshow("outvsOUT", np.hstack((out,OUT)))
    #cv.imshow("OPENINGvsBinOUT", np.hstack((opening, bin_OUT)))
    #cv.imshow("SUREFGvsBinOUT", np.hstack((sure_fg, bin_OUT)))
   # cv.imshow("SUREBGvsBINOUT", np.hstack((sure_bg, bin_OUT)))
    # cv.imshow("UNKnown",unknown)
    cv.imshow("last",tmp)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()

