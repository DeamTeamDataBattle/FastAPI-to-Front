#!/bin/python3
import os, sys, cv2, numpy as np, time, glob, pypdfium2 as pdfium, matplotlib.pylab as plt, pytesseract, matplotlib
from PIL import Image

def group_pt(loc, H):
    print(len(loc))
    pts = []
    i = 0
    while i < len(loc)-3:
        start = loc[i][1]
        while loc[i][1] <= loc[i+1][1] <= loc[i][1] + H:
            i += 1
            if i >= len(loc)-1:
                break
        pts.append([start, loc[i-1][1]])
        i += 1


    return pts

# get file pdf
if len(sys.argv) > 2:
    pdf_path = sys.argv[sys.argv.index('-f')+1]
else:
    print("no image given")
    sys.exit()
print(pdf_path)

dir_path = "data/"+pdf_path[5:-4]

print(dir_path)

if not os.path.exists(dir_path):
    raise Exception("please run script first")


files = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f[-3:] == "jpg" and "pattern" not in f and "log" not in f and "legend" not in f]
images = [cv2.imread(f) for f in files]

log_image_path = os.path.join(dir_path, "log_1.jpg")
log_image = cv2.imread(log_image_path)
H,W,D = log_image.shape
img = cv2.cvtColor(log_image, cv2.COLOR_BGR2GRAY)
#img = cv2.GaussianBlur(img, (5,5),0)

clay_template = [cv2.imread(f,0) for f in files if "clay" in f][0]

thresh = 0.3
w, h = clay_template.shape[::-1]
if W > w:
    res = cv2.matchTemplate(img, clay_template, cv2.TM_CCOEFF)
    loc = np.where( res >= thresh )
    pts = group_pt(list(zip(*loc[::-1])), h)
    for y1, y2 in pts:
        log_image = cv2.rectangle(log_image, [10,y1], [10+w,y2] , (0, 0, 255), 5)

sand_template = [cv2.imread(f,0) for f in files if "sand" in f][0]

thresh = 0.16
w, h = sand_template.shape[::-1]
if W > w:
    res = cv2.matchTemplate(img, sand_template, cv2.TM_CCOEFF)
    loc = np.where( res >= thresh )
    pts = group_pt(list(zip(*loc[::-1])), h)
    for y1, y2 in pts:
        log_image = cv2.rectangle(log_image, [20,y1], [20+w,y2] , (255, 0, 0), 5)


cv2.imwrite(dir_path+"/log_PM.jpg", log_image)


