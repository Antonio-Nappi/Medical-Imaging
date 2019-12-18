import cv2 as cv
import numpy as np
import math
filepath = "dataset\images\mass\\20586908_6c613a14b80a8591_MG_R_CC_ANON.tif"

def normalize(img):
    tmp = img-np.amin(img)
    image = tmp/np.amax(img)
    return image

def regional_mean(img,list):
    tmp = cv.blur(img,(list[0],list[1]))
    return cv.resize(tmp, (img.shape[0],img.shape[1]),interpolation=cv.INTER_LINEAR)

def normalize_and_HE(img):
    t = cv.equalizeHist(np.asarray(normalize(img),dtype=np.uint8))
    return t

img = cv.imread(filepath,cv.IMREAD_GRAYSCALE)

img2 = normalize(img)

clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img_clahe = clahe.apply(img)
mu = np.average(img)
tmp = normalize(img_clahe)
img3 = np.zeros((tmp.shape[0],tmp.shape[1]))
for i in range(img_clahe.shape[0]):
    for j in range(img_clahe.shape[1]):
        img3[i,j]= tmp[i,j] * (1-math.exp(-(img2[i,j]/mu)))
img3 = normalize(img3)
m2 = regional_mean(img3,[16,16])
img4 = normalize_and_HE(img3)
img5 = normalize(img3 + regional_mean(m2,[16,16]))
img6 = normalize_and_HE(img5) * img4
e = normalize(img6)
#cv.warpPolar(e,p,)
while 1:
    cv.imshow("Immagine di partenza",cv.resize(img,(500,500)))
    cv.imshow("Img2", cv.resize(img2,(500,500)))
    cv.imshow("Img_clahe",cv.resize(img_clahe,(500,500)))
    cv.imshow("img3", cv.resize(img3,(500,500)))
    cv.imshow("img4", cv.resize(img4,(500,500)))
    cv.imshow("img5",cv.resize(img5,(500,500)))
    cv.imshow("img6",cv.resize(img6,(500,500)))
    # cv.imshow("stretchVSLog", np.hstack((str_img_r,out)))
    # cv.imshow("imageVSLog", np.hstack((start, OUT)))
    # cv.imshow("outvsOUT", np.hstack((out,OUT)))
    # cv.imshow("OPENINGvsBinOUT", np.hstack((opening, bin_OUT)))
    # cv.imshow("SUREFGvsBinOUT", np.hstack((sure_fg, bin_OUT)))
    # cv.imshow("SUREBGvsBINOUT", np.hstack((sure_bg, bin_OUT)))
    # cv.imshow("UNKnown",unknown)
    # cv.imshow("Img",img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()