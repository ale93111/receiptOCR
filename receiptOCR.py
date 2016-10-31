# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 01:26:38 2016

@author: alessandro
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:35:53 2016

@author: alessandro
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import string

path = '/home/alessandro/bioretics/'

def get_big_cont(contours):
    areacnt = []
    for c in contours:
         areacnt.append(cv2.contourArea(c))
         
    return contours[areacnt.index(max(areacnt))]
    

def resize_win(img2,t):
    #kernel = np.array([[0,0,0],[0,1,1],[0,0,0]], np.uint8)
    #dilation = cv2.dilate(img2,kernel,iterations = m)
    #split image in half and apply opposite dilation morphology
    #to avoid overstep image boundaries
    img_sx = img2.copy()
    img_dx = img2.copy()
    wimg, himg = img2.shape    
    img_dx[:,0:int(himg/2)]  = 0
    img_sx[:,int(himg/2):-1] = 0
    kernelsx = np.array([[0,0,0],[0,1,1],[0,0,0]], np.uint8)
    kerneldx = np.array([[0,0,0],[1,1,0],[0,0,0]], np.uint8)

    m = find_dx(img2)   
    dilationsx = cv2.dilate(img_sx,kerneldx,iterations = 2*m)
    dilationdx = cv2.dilate(img_dx,kernelsx,iterations = 2*m)
    dilation = dilationsx | dilationdx

    _, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    x,y,w,h = cv2.boundingRect( get_big_cont(contours) )
    
    img3 = cv2.copyMakeBorder(img2[(y):(y+h),(x):(x+w)],t,t,t,t,cv2.BORDER_CONSTANT,0)
    plt.imshow(img3, cmap='gray')   
    
    return img3.copy()
    
def net_predict_label(net, input_image):
    #immagine a livelli di grigio
    input_image = input_image/255
    input_image = np.expand_dims(input_image,2)
    #input_image=input_image.astype(np.uint8)
    #print (input_image.shape, input_image.dtype)
    
    prediction = net.predict([input_image],oversample=False) 
    print(prediction)
    return np.argmax(prediction)

def remove_leading_01(array):
    i = 1
    if array[0] == 0:
        while array[i] != 1:
            i = i + 1
        
        return array[i-5:-1]
        
    else:
        while array[i] != 0:
            i = i + 1
        
        return array[i:-1]
        
def remove_ending_01(array):
    i = np.size(array) - 1
    if array[-1] == 0:
        while array[i] != 1:
            i = i - 1
        
        return array[0:i+5]
        
    else:
        while array[i] != 0:
            i = i - 1
        
        return array[0:i]
def find_avgind0(array):
    numc = []
    i = 1
    c1 = 0
    c2 = 0
    while i<len(array)-1:
        if( array[i] == 0 ):
            c1 += i
            c2 += 1
            
        if(array[i] == 1 and array[i-1] == 0):
            if c2 is not 0:
                numc.append(c1/c2)
            c1 = 0
            c2 = 0
        
        i += 1
    numc.append(len(array)-1)
    numc = [int(i) for i in numc]
    return numc

def find_dx(img):
    sumx0 = np.sum(img, axis=0)>0
    #numc = find_avgind0(sumx0)
    
    array1 = np.diff(sumx0)
    indpos = np.where(array1==True)
    #indd = [sum(indpos[0][current: current+2]) for current in range(0, len(indpos[0]), 2)]
    #magic - find width of projected elements
    indd = [t - s for s, t in zip(indpos[0], indpos[0][1:])][::2]
    #inddm = np.delete(indd,np.argmax(indd))
    
    dx = np.average(indd)
    return int(dx)

def find_dy(img):
    sumy0 = np.sum(img, axis=1)>0

    array1 = np.diff(sumy0)
    indpos = np.where(array1==True)
    #indd = [sum(indpos[0][current: current+2]) for current in range(0, len(indpos[0]), 2)]
    #magic - find width of projected elements
    indd = [t - s for s, t in zip(indpos[0], indpos[0][1:])][::2]
    #inddm = np.delete(indd,np.argmax(indd))
    
    dy = np.max(indd)

    return dy
    
def sanitize(pr):
    safepr = pr.copy()
    dy = []
    for i,im in enumerate(safepr):
        dy.append(find_dy(im))

    tresh = 0.15
    listdel = []
    for j,dyj in enumerate(dy):
        if dyj < (1-tresh)*np.median(dy) or dyj > (1+tresh)*np.median(dy):
            listdel.append(j)
            
    for k in listdel[::-1]:
        del safepr[k]
            
    return safepr
    
#%%
img = cv2.imread(path+'s10.jpg', 0)
#ret, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
otsu = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
otsu = 255 - otsu

# Some morphology to clean up image
kernel = np.ones((8,8), np.uint8)
opening = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
otsu    = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 12))
plt.imshow(img, cmap='gray')

himg, wimg = img.shape
xmargin = int(0.06*wimg)
ymargin = int(0.06*himg)
t = 5

#%%
plt.figure(figsize=(12, 12))
plt.imshow(otsu, cmap='gray')
#%%
#TOTposx = 738 #per s5
#TOTposy = 817 #per s5
#TOTposx = 478 #per s4
#TOTposy = 912 #per s4
#TOTposx = 850 #per s2
#TOTposy = 625 #per s2
#TOTposx = 723 #per s1
#TOTposy = 850 #per s1
TOTposx = 3085 #per s10
TOTposy = 2589 #per s10
#TOTposx = 2304 #per s11
#TOTposy = 2616 #per s11
#TOTposx = 2086 #per s12
#TOTposy = 2673 #per s12
#%%
width, height = img.shape
tot = otsu[(TOTposx-xmargin-t):(TOTposx+xmargin+t), 
           (TOTposy-ymargin-t):(TOTposy+ymargin+t)]

plt.figure(figsize=(7,7))
plt.imshow(tot, cmap='gray')
#%%
cif = resize_win(tot,t)
#%%
im = Image.fromarray(np.uint8(plt.cm.gist_earth(cif)*255))
totprice = pytesseract.image_to_string(im) 
totprice = totprice.replace(" ","")
totprice = totprice.replace(",",".") 

print("totprice = ", totprice)
#%%
allpr = otsu[(0):(TOTposx+xmargin), 
           (TOTposy-ymargin):(TOTposy+ymargin)]

plt.figure(figsize=(7,7))
plt.imshow(allpr,cmap='gray')

#%%
sumxallpr = np.sum(allpr, axis=1)
sumxallpr0 = sumxallpr>0.05*np.max(sumxallpr)

#remove leading or ending zeros or ones
#sumxallpr0 = remove_leading_01(sumxallpr0)
sumxallpr0 = remove_ending_01(sumxallpr0)

plt.figure(figsize=(7, 7))
plt.plot(sumxallpr0)
#%%
numpr = find_avgind0(sumxallpr0)
print(numpr)

#segmenting numbers based on pre-computed grid positions
pr = []
for i in range(len(numpr)-1):
    pr.append(allpr[numpr[i]:numpr[i+1],:])
#%%
plt.figure(figsize=(7, 7))
plt.imshow(pr[0], cmap='gray')
#%%
for impr in pr:
    plt.imshow(impr, cmap='gray')
    print("dy = ", find_dy(impr))
    plt.show()
#%%
safepr = sanitize(pr)

for impr in safepr:
    plt.imshow(impr, cmap='gray')
    print("dy = ", find_dy(impr))
    plt.show()
#%%
prcif = []
for impr in safepr:
    prcif.append(resize_win(impr,t))

for imprcif in prcif:
    plt.imshow(imprcif, cmap='gray')
    #print("dy = ", find_dy(imprcif))
    plt.show()
#%%    
pricelist = []

for impr in prcif:    
    im = Image.fromarray(np.uint8(plt.cm.gist_earth(impr)*255))
    pricelist.append(pytesseract.image_to_string(im) )

print(pricelist)

#%%
#remove spaces, letters and \n
complete_list = string.ascii_letters + ' ' + '\n'
for j,pricej in enumerate(pricelist):
    pricej = pricej.replace(",",".")    
    pricelist[j] = pricej.translate({ord(c): None for c in complete_list})

#remove empty strings   
while '' in pricelist: pricelist.remove('')

print("pricelist = ", pricelist)
#%%
npricelist = []
npricetot  = []
for pricej in pricelist:
    for t in pricej.split():
        try:
            npricelist.append(float(t))
        except ValueError:
            pass
for t in totprice.split():
    try:
        npricetot.append(float(t))
    except ValueError:
        pass

print("price list  = ", npricelist)
print("total price = ", npricetot)

if npricetot[0] == np.sum(npricelist):
    print("\n","Everything's OK!")
else: 
    print("\n","Please check prices")
#%%
import cv2
import numpy as np

ix,iy = -1,-1
# mouse callback function
def get_posxy(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix,iy = x,y

# Create a window and bind the function to window
cv2.namedWindow('image',cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback('image',get_posxy)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('a'):
        print (ix,iy)
cv2.waitKey(1)    
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)

TOTposy = ix
TOTposx = iy
    