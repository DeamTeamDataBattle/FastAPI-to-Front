import os, sys, cv2, numpy as np, time, glob, pypdfium2 as pdfium, matplotlib.pylab as plt, pytesseract, matplotlib, torch, json
from PIL import Image
from scripts.get_log_page import get_log_image 

def write_notif(notif, write=True):
    if not write:
        text = open("data/notification.txt", "r").read()
    else:
        text = ""
    with open("data/notification.txt", "w") as f:
        f.write(text+notif)

LABELS = ["legend", "log", "pattern"]

def map_coords(x,y,w_scl,h_scl):
    return int(x*w_scl), int(y*h_scl)

def separate_pattern(img, path):
    image = np.array(img)
    H,W,D = image.shape

    image_text = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_text = cv2.threshold(image_text, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image_text = cv2.bitwise_not(image_text)
    kernel = np.ones((2, 1), np.uint8)
    image_text = cv2.dilate(image_text, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    image_text = cv2.erode(image_text, kernel, iterations=1)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.threshold(img, 200, 200, cv2.THRESH_BINARY_INV)[1]
    img = cv2.dilate(img, np.ones((1, 3)))
    #img = cv2.GaussianBlur(img, (2, 2), 0)
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #image = cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
        if h > 0.5 * H and w < 0.7 * W:
            box = [x,y,w,h]
            break
            #image = cv2.rectangle(image, (x,y), (x+w, y+h), color=(0,0,255), thickness=1)
        elif w > 0.2 * W:
            text = [x,y,w,h]
        #all_text += [''.join(filter(str.isalpha, s)) for s in pytesseract.image_to_string(crop_img).lower().replace("\n", "").split(" ") if len(s) > 4]

    #cv2.imwrite(path.format(""), image)
    #cv2.imwrite(path.format("con"), image)
    #cv2.imwrite(path.format("mod"), img)
    #cv2.imwrite(path.format("text_"+text), image_text)
    if box:
        text = pytesseract.image_to_string(image_text)
        words = [c for c in text.lower().split("\n") if len(c) > 3]
        if words:
            word = ''.join([s for s in words[0].split(" ") if len(s) > 3])
            text = ''.join([s for s in filter(str.isalpha, word)])
            x,y,w,h = box
            s = 5
            box_img = image[y+s:y+h-s, x+s:x+w-s]
            cv2.imwrite(path.format(text), box_img)

def separate_log(coords, image, path):
    img = np.array(image)
    H,W,D = img.shape
    Xmin,Xmax,Ymin,Ymax = [],[],[],[]
    for xmin,ymin,xmax,ymax in coords:
        Xmin.append(xmin)
        Ymin.append(ymin)
        Xmax.append(xmax)
        Ymax.append(ymax)

    X_min = int(sum(Xmin) / len(Xmin))  
    #Y_min = int(sum(Ymin) / len(Ymin))
    X_max = int(sum(Xmax) / len(Xmax))
    #Y_max = int(sum(Ymax) / len(Ymax))
    Y_min = 0
    Y_max = H

    w = 20
    log_img = img[Y_min:Y_max, X_min-w:X_max+w]
    H, W, D = log_img.shape
    log_img_1 = log_img[:H//2]
    log_img_2 = log_img[H//2:]
    cv2.imwrite(path.format("1"), log_img_1)
    cv2.imwrite(path.format("2"), log_img_2)



"""
main script to run to get legend + patterns
"""

def process(pdf_path):

    # import model from trained weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='scripts/weights.pt')
    print('model loaded')
    write_notif("model loaded")

    dir_path = "data/images/"+pdf_path[10:-4]

    print(dir_path)
    print(pdf_path)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    log_image = get_log_image(pdf_path, dpi=200, save=False)
    cv2.imwrite(dir_path+"/log_large.jpg", log_image)
    write_notif("log page found")

    # NN trained on img 320
    img = Image.fromarray(log_image)
    img_size = 320
    w,h = img.size
    W,H = img_size, int(img_size/w*h)
    width_scl = w / W
    height_scl = h / H
    resized_image = img.resize((W,H))
    i = 0
    log_coords = []
    while True:
        write_notif("scanning log page %d" % i)
        end = False
        if (i+1)*img_size > H:
            height = H
            end = True
        else:
            height = (i+1)*img_size
        cropped_small = resized_image.crop((0, i*img_size, W, height))
        if (i+1)*w > h:
            height = h
            end = True
        else:
            height = (i+1)*w 
        cropped_large = img.crop((0, i*w, w, height))
        if not end:
            i += 1
        else:
            break
        results = model(cropped_small)
        res = json.loads(results.pandas().xyxy[0].to_json())
        if res["name"]:
            for key, value in res["class"].items():
                xmin = int(res["xmin"][key])
                ymin = int(res["ymin"][key])
                xmax = int(res["xmax"][key])
                ymax = int(res["ymax"][key])
                crop_small = cropped_small.crop((xmin, ymin, xmax, ymax))
                #crop_small.save(dir_path+"/%s_%s_small.jpg" % (LABELS[value], key))
                xmin, ymin = map_coords(xmin, ymin, width_scl, height_scl)
                xmax, ymax = map_coords(xmax, ymax, width_scl, height_scl)
                crop_large = cropped_large.crop((xmin, ymin, xmax, ymax))
                #crop_large.save(dir_path+"/%s_%s_large.jpg" % (LABELS[value], key))
                if value == 2:
                    separate_pattern(crop_large, dir_path+"/{}.jpg")
                if value == 1:
                    log_coords.append([xmin, ymin, xmax, ymax])
                    #crop_large.save(dir_path+"/%d_%s_log.jpg" % (i, key))
                if value == 0:
                    print("legend found", i)
                    crop_large.save(dir_path+"/%d_legend.jpg" % i)

    separate_log(log_coords, img, dir_path+"/log_{}.jpg")
    return {"info": "finished :D"}
