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
import tesserocr
import string

path = '/home/alessandro/bioretics/'

def get_big_cont(contours):
    areacnt = []
    for c in contours:
         areacnt.append(cv2.contourArea(c))
         
    return contours[areacnt.index(max(areacnt))]
    

def resize_win(img2,t,returnpos=False):
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
    if returnpos:
        return x,y,h,w
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

def segment_cif(img,t):
    temp = resize_win(img,t)
    
    sumx0 = np.sum(temp, axis=0)>0    

    #find averages of index positions of intervals where 
    #img projection on x-axis is 0 to evaluate grid spacing
    numc = find_avgind0(sumx0)
    #print(numc)
    
    #segmenting numbers based on pre-computed grid positions
    cif = []
    for i in range(len(numc)-1):
        cif.append(temp[:,numc[i]:numc[i+1]])
        
    return cif
    
def find_dx(img):
    sumx0 = np.sum(img, axis=0)>0
    #numc = find_avgind0(sumx0)
    
    array1 = np.diff(sumx0)
    indpos = np.where(array1==True)
    #indd = [sum(indpos[0][current: current+2]) for current in range(0, len(indpos[0]), 2)]
    #magic - find width of projected elements
    indd = [t - s for s, t in zip(indpos[0], indpos[0][1:])][::2]
    #inddm = np.delete(indd,np.argmax(indd))
    
    if not indd:
        print("cannot find dx, empty index list")
        return 0
    else:
        dx = np.max(indd)
        
    return dx

def find_dy(img):
    sumy0 = np.sum(img, axis=1)>0

    array1 = np.diff(sumy0)
    indpos = np.where(array1==True)
    #indd = [sum(indpos[0][current: current+2]) for current in range(0, len(indpos[0]), 2)]
    #magic - find width of projected elements
    indd = [t - s for s, t in zip(indpos[0], indpos[0][1:])][::2]
    #inddm = np.delete(indd,np.argmax(indd))
    if not indd:
        print("cannot find dy, empty index list")
        return 0
    else:
        dy = np.max(indd)

    return dy
    
def sanitize(img_prices):
    safe_img_prices = img_prices.copy()
    dy = []
    for i,im in enumerate(safe_img_prices):
        dy.append(find_dy(im))

    tresh = 0.15
    listdel = []
    for j,dyj in enumerate(dy):
        if dyj < (1-tresh)*np.median(dy) or dyj > (1+tresh)*np.median(dy):
            listdel.append(j)
            
    for k in listdel[::-1]:
        del safe_img_prices[k]
            
    return safe_img_prices
    
def stringtofloat(listin):
    result = []    
    for t in listin.split():
        try:
            result.append(float(t))
        except ValueError:
            pass
    if not result:
        return 0
    else:
        return result[0]
    
def recognize_price(api, im_price):
    im = Image.fromarray(np.uint8(plt.cm.gist_earth(im_price)*255))
    api.SetImage(im)
    api.Recognize() 
    
    return api.GetUTF8Text(), api.AllWordConfidences()  

def isequal(a, b, rel_tol=1e-05, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
#%%
#TESSERACT
api = tesserocr.PyTessBaseAPI(psm=8,init=True)
#api.InitFull(variables=dict(load_system_dawg="0"))
api.SetVariable("tessedit_char_whitelist", "0123456789.,-")

img_receipt = cv2.imread(path+'s10.jpg', 0)

#ret, img_binary = cv2.threshold(img_receipt,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
img_binary  = cv2.adaptiveThreshold(img_receipt, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 61, 3)
#img_binary = 255 - img_binary

# Some morphology to clean up image
kernel     = np.ones((8,8), np.uint8)
opening    = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)
img_binary = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#img_binary = opening.copy()

plt.figure(figsize=(12, 12))
plt.imshow(img_receipt, cmap='gray')

himg, wimg = img_receipt.shape
xmargin = int(0.06*wimg)
ymargin = int(0.06*himg)
t = 5

#%%
plt.figure(figsize=(12, 12))
plt.imshow(img_binary, cmap='gray')
#%%
TOTposx = 3485 #per s10
TOTposy = 2589 #per s10 OK
#TOTposx = 2304 #per s11
#TOTposy = 2616 #per s11 OK
#TOTposx = 2086 #per s12
#TOTposy = 2673 #per s12 OK
#TOTposx = 2923 #per s13
#TOTposy = 2695 #per s13 #6-8 x2
#TOTposx = 3200 #per s14 15 #OK
#TOTposy = 2650 #per s14 15 #noise+sconto
#TOTposx = 3780 #per s16
#TOTposy = 2700 #per s16 #OK
#%%
width, height = img_receipt.shape
img_tot = img_binary[(TOTposx-xmargin-t):(TOTposx+xmargin+t), 
                     (TOTposy-ymargin-t):(TOTposy+ymargin+t)]

plt.figure(figsize=(7,7))
plt.imshow(img_tot, cmap='gray')
#%%
win_tot = resize_win(img_tot,t)
x,y,h,w = resize_win(img_tot,t,returnpos=True)
#%%
totprice, _ = recognize_price(api,win_tot)

totprice = totprice.replace(" ","")
totprice = totprice.replace(",",".") 

print("totprice = ", totprice)
#%%
img_allprices = img_binary[(0):(TOTposx+xmargin-h-y), 
                           (TOTposy-ymargin):(TOTposy+ymargin)]

plt.figure(figsize=(7,7))
plt.imshow(img_allprices,cmap='gray')

#%%
sumxallpr = np.sum(img_allprices, axis=1)
sumxallpr0 = sumxallpr>0.05*np.max(sumxallpr)

#remove leading or ending zeros or ones
#sumxallpr0 = remove_leading_01(sumxallpr0)
sumxallpr0 = remove_ending_01(sumxallpr0)

plt.figure(figsize=(7, 7))
plt.plot(sumxallpr0)
#%%
indexpos0 = find_avgind0(sumxallpr0)
print(indexpos0)

#segmenting numbers based on pre-computed grid positions
img_prices = []
for i in range(len(indexpos0)-1):
    img_prices.append(img_allprices[indexpos0[i]:indexpos0[i+1],:])
#%%
plt.figure(figsize=(7, 7))
plt.imshow(img_prices[0], cmap='gray')
#%%
for im_price in img_prices:
    #plt.imshow(im_price, cmap='gray')
    print("dy = ", find_dy(im_price))
    plt.show()
#%%
safe_img_prices = sanitize(img_prices)

for im_price in safe_img_prices:
    #plt.imshow(im_price, cmap='gray')
    print("dy = ", find_dy(im_price))
    plt.show()
#%%
win_prices = []
for im_price in safe_img_prices:
    win_prices.append(resize_win(im_price,t))

for im_price in win_prices:
    plt.imshow(im_price, cmap='gray',interpolation=None)
    #print("dy = ", find_dy(im_price))
    plt.show()
#%%
#correction removing letters
api.SetVariable("tessedit_char_whitelist", string.ascii_uppercase+"0123456789.,-")
delist    = []
for j,im_price in enumerate(win_prices):        
    text, prob = recognize_price(api,im_price)
    if not any(x in text for x in string.digits):    
        delist.append(j)
    #problist.append(prob)
    #print(text, prob)

for j in delist[::-1]:        
    del win_prices[j]
#%%    
api.SetVariable("tessedit_char_whitelist", "0123456789.,-")
pricelist = []
delist    = []
for j,im_price in enumerate(win_prices):        
    text, prob = recognize_price(api,im_price)
    if prob[0]>50:
        pricelist.append( text)
    else: 
        delist.append(j)
    #problist.append(prob)
    print(prob)

print(pricelist)

for j in delist[::-1]:        
    del win_prices[j]
    
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
#Error correction
#correction of prices that don't contain '.'
#errprindex = []
#for j,pricej in enumerate(pricelist):
#    if '.' not in pricej:
#        errprindex.append(j)
#        
errprindex = [j for j,pricej in enumerate(pricelist) if '.' not in pricej]

print(errprindex)

template = cv2.imread(path+'modelv.jpg', 0)

err_prices = [segment_cif(win_prices[j],t) for j in errprindex]

#resize every digit for template matching
for i,pr in enumerate(err_prices):
    for j,im in enumerate(pr):
        err_prices[i][j] = cv2.resize(im, (28,28))

temp_res = []
for i,pr in enumerate(err_prices):
    res = []    
    for j,im in enumerate(pr):
        res.append(cv2.matchTemplate(im,template,cv2.TM_CCOEFF))
    temp_res.append(res)

err_pos = [np.argmax(i) for i in temp_res]

for j,i in enumerate(errprindex):
    list1 = list(pricelist[i])
    list1[err_pos[j]] = '.'
    pricelist[i] = ''.join(list1)

#%%
npricelist = [stringtofloat(pricej) for pricej in pricelist]
npricetot  = stringtofloat(totprice)

print("price list  = ", npricelist)
print("total price = ", npricetot)
ok = False
if isequal(npricetot,np.sum(npricelist)):
    print("\n","Everything's OK!")
    ok = True
else: 
    print("\n","Please check prices")
    ok = False
#%%
b = []
if not ok:
    a = np.argmax(npricelist)
    b = npricelist.copy()
    del b[a:]
    if isequal(npricelist[a],np.sum(b)):
        print("removed subtotal")
        del npricelist[a]
        
print("price list  = ", npricelist)
print("total price        = ", npricetot)
print("np.sum(npricelist) = ", np.sum(npricelist))

if isequal(npricetot,np.sum(npricelist)):
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
    cv2.imshow('image',img_receipt)
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
    