import os, sys, cv2, numpy as np, time, glob, pypdfium2 as pdfium, matplotlib.pylab as plt, pytesseract, matplotlib, torch, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
from scripts.get_log_page import get_log_image
from scripts.extract import cluster_log
from scripts.functions import write_notif, LOG_COLUMN_WIDTH
#from tyFinder.decoupageImage import finale

LABELS = ["legend", "log", "pattern"]

def map_coords(x,y,w_scl,h_scl):
    return int(x*w_scl), int(y*h_scl)

def resize_patterns(pattern_dir):
    patterns = [os.path.join(pattern_dir, f) for f in os.listdir(pattern_dir)]
    if len(patterns) == 0:
        raise Exception("No patterns found.. ")
    pattern_imgs_orig = [cv2.imread(os.path.join(pattern_dir, f)) for f in os.listdir(pattern_dir)]
    for i in range(len(pattern_imgs_orig)):
        img = pattern_imgs_orig[i]
        h,w,d = img.shape
        img = cv2.resize(img, (LOG_COLUMN_WIDTH, int(LOG_COLUMN_WIDTH/w*h)))
        pattern_imgs_orig[i] = img
    min_height = min(i.shape[0] for i in pattern_imgs_orig)
    for i in range(len(pattern_imgs_orig)):
        img = pattern_imgs_orig[i]
        h,w,d = img.shape
        y1 = h//4
        y2 = y1 + min_height
        if y2 >= h:
            y1 = h - min_height
            y2 = h
        img = img[y1:y2]
        cv2.imwrite(patterns[i], img)
    print("resized")

def check_legend(img, path):
    image_text = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_text = cv2.threshold(image_text, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((2, 1), np.uint8)
    #image_text = cv2.erode(image_text, kernel, iterations=2)
    image_text = cv2.dilate(image_text, kernel, iterations=1)
    #cv2.imwrite(path.format("text"), image_text)
    text = pytesseract.image_to_string(image_text).lower()
    if "lithological" in text or "legend" in text:
        return True
    return False

def extract_patterns(image, path):
    H, W, D = image.shape
    image_text = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_text = cv2.threshold(image_text, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    image_text = cv2.bitwise_not(image_text)
    kernel = np.ones((2, 1), np.uint8)
    #image_text = cv2.erode(image_text, kernel, iterations=1)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 3
    image_text = cv2.filter2D(image_text, -1, kernel)
    kernel = np.ones((1, 1), np.uint8)
    image_text = cv2.erode(image_text, kernel, iterations=1)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.threshold(img, 200, 200, cv2.THRESH_BINARY_INV)[1]
    img = cv2.dilate(img, np.ones((1, 3)))
    #img = cv2.GaussianBlur(img, (2, 2), 0)
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        #image = cv2.rectangle(image, (x,y), (x+w, y+h), color=(255,0,0), thickness=1)
        if 1.3 < w / h < 2.7 and h > 40:
            boxes.append([x,y,w,h])
            #image = cv2.rectangle(image, (x,y), (x+w, y+h), color=(0,0,255), thickness=2)
    
    #cv2.imwrite(path.format("con"), image)
    #cv2.imwrite(path.format("mod"), img)
    #cv2.imwrite(path.format("text"), image_text)
    textes = []
    s = 5
    for bx, by, bw, bh in boxes:
        names = []
        crop_img = image[min(by+s,H):max(by+bh-s,0),min(bx+s,W):max(bx+bw-s,0)]
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if by < y < by+bh and bx < x < bx+bw*2:
                crop_text = image_text[max(y-s,0):min(y+h+s,H),max(x-s,0):min(x+w+s,W)]
                text = pytesseract.image_to_string(crop_text)
                words = [c for c in text.lower().split("\n") if len(c) > 1]
                if words:
                    word = ''.join([s for s in words[0].split(" ") if len(s) > 1])
                    names.append(''.join([s for s in filter(str.isalpha, word)]))
        if names:
            name = max(names, key=len)
            cv2.imwrite(path.format("legend_"+name), crop_img)

def find_log_legend(image, dir_path, pattern_dir):
    image = np.array(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = cv2.threshold(img, 200, 200, cv2.THRESH_BINARY_INV)[1]
    #img = cv2.erode(img, np.ones((1, 20)))
    img = cv2.dilate(img, np.ones((1, 5)))
    img = cv2.GaussianBlur(img, (3, 3), 0)

    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imgs = []
    s = 10
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        crop_img = image[y+s:y+h-s,x+s:x+w-s]
        imgs.append(crop_img)

    #cv2.imwrite(path.format("legend_p"), image)

    for img in imgs:
        if check_legend(img, dir_path):
            #cv2.imwrite(path.format("legend"), img)
            extract_patterns(img, pattern_dir+"/{}.jpg")
        else:
            H,W,D = img.shape
            log_img = cv2.resize(img, (LOG_COLUMN_WIDTH, int(LOG_COLUMN_WIDTH/W*H)))
            cv2.imwrite(dir_path+"/log_image.jpg", log_img)
    
    return dir_path+"/log_image.jpg"

# log extration
# the aim is to find the log colon and straighten it
def separate_log(coords, image, path):
    # img is PIL
    # I really don't want to change this since it works okish
    img = np.array(image)
    H,W,D = img.shape
    # crop large log image to just where it found the log column
    Xmin_arr,Xmax_arr,Ymin_arr,Ymax_arr = [],[],[],[]
    for xmin,ymin,xmax,ymax in coords:
        Xmin_arr.append(xmin)
        Ymin_arr.append(ymin)
        Xmax_arr.append(xmax)
        Ymax_arr.append(ymax)

    X_min = int(sum(Xmin_arr) / len(Xmin_arr))  
    #Y_min = int(sum(Ymin_arr) / len(Ymin_arr))
    X_max = int(sum(Xmax_arr) / len(Xmax_arr))
    #Y_max = int(sum(Ymax_arr) / len(Ymax_arr))
    Y_min = 0
    Y_max = H
    w = (X_max-X_min)*2
    log_img = img[Y_min:Y_max, X_min-w:X_max+w]
    H, W, D = log_img.shape
    # resize to 75px width
    WIDTH = LOG_COLUMN_WIDTH*5
    log_img = cv2.resize(log_img, (WIDTH, int(WIDTH/W*H)))
    H, W, D = log_img.shape
    N = H//W
    widths = []
    x_start = []
    points = []
    # split into smaller images and extract
    for i in range(N):
        log_img_crop = log_img[i*H//N:min((i+1)*H//N, H)]
        img_h, img_w, img_d = log_img_crop.shape
        # straighten image
        log_img_crop_mod = cv2.cvtColor(log_img_crop, cv2.COLOR_BGR2GRAY)
        log_img_crop_mod = cv2.GaussianBlur(log_img_crop_mod, (3,3), 0)
        log_img_crop_mod = cv2.threshold(log_img_crop_mod, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Detect vertical line
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,W//5))
        log_img_crop_mod = cv2.morphologyEx(log_img_crop_mod, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        contours, hierarchy = cv2.findContours(log_img_crop_mod,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        xs = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if h > W//6:
                xs.append(x)
                #log_img_crop = cv2.rectangle(log_img_crop, (x,0), (x+w, W), color=(255,0,0), thickness=1)

        xs.sort()
        pairs = []
        gaps = []
        for j in range(len(xs)-1):
            g = abs(xs[j] - xs[j+1])
            if img_w//6 < g < img_w//3:
                #log_img_crop = cv2.rectangle(log_img_crop, (xs[j],0), (xs[j+1], W), color=(255,255,0), thickness=1)
                pairs.append([xs[j], xs[j+1]])
                widths.append(np.ceil(g/2)*2)
                #gaps.append(np.ceil(g/2)*2)

        found = False
        if widths:
            gap = int(max(set(widths), key=widths.count))
        #if gaps:
            #gap = int(max(set(gaps), key=gaps.count))
            widths.append(gap)
            for x1, x2 in pairs:
                x1=min(x1,x2)
                x2=max(x1,x2)
                g = abs(x2 - x1)
                if 0.9*gap < g < 1.1*gap:
                    # right size
                    if len(x_start) > 0:
                        x1_t = x_start[-1]
                        if not (x1 - 15 < x1_t < x1 + 15):
                            x1 = x1_t
                        if len(xs) > 0:
                            X = x1
                            d_min = W
                            for x in xs:
                                d = abs(x - x1)
                                if d < gap*2 and d < d_min:
                                    d_min = d
                                    X = x
                            x1 = X
                    x2 = x1 + gap
                    x_start.append(min(x1, x2))
                    points.append([min(x1,x2),i*H//N])
                    #log_img_crop = cv2.rectangle(log_img_crop, (x1,0), (x2, W), color=(0,0,255), thickness=2)
                    found = True
                    break

        if not found and len(x_start) > 0:
            # use last value closest to last
            gap = int(max(set(widths), key=widths.count))
            x1 = x_start[-1]
            if len(xs) > 0:
                X = x1
                d_min = W
                for x in xs:
                    d = abs(x - x1)
                    if d < gap/2 and d < d_min:
                        d_min = d
                        X = x
                x1 = X

            x2 = x1 + gap
            #log_img_crop = cv2.rectangle(log_img_crop, (x1,0), (x2, W), color=(0,255,0), thickness=2)
            x_start.append(x1)
            points.append([min(x1,x2),i*H//N])

        #cv2.imwrite(path.format(str(i)), log_img_crop)
        #cv2.imwrite(path.format(str(i)+"_mod"), log_img_crop_mod)

    if len(widths) == 0 or len(points) == 0:
        raise Exception("Log column not found")

    g = int(max(set(widths), key=widths.count))
    x1, y1 = points[0]
    img1 = log_img[y1:y1+W, x1:x1+g]
    for i in range(1, len(points)-1):
        x1, y1 = points[i]
        if x1+g >= W:
            x1 = W - g
        img2 = log_img[y1:y1+W, x1:x1+g]
        img1 = np.concatenate((img1, img2), axis=0)
    H,W,D = img1.shape
    img1 = cv2.resize(img1, (LOG_COLUMN_WIDTH, int(LOG_COLUMN_WIDTH/W*H)))
    cv2.imwrite(path.format("image"), img1)
    return path.format("image")

def crop_image_square(img, i, w, h):
    # cropped image is square so w is h
    height = min((i+1)*w, h)
    return img.crop((0, i*w, w, height)), h==height

def extract_image_from_res(img, res, key, width_scl=0,height_scl=0):
    xmin = int(res["xmin"][key])
    ymin = int(res["ymin"][key])
    xmax = int(res["xmax"][key])
    ymax = int(res["ymax"][key])
    if width_scl != 0 and height_scl != 0:
        xmin, ymin = map_coords(xmin, ymin, width_scl, height_scl)
        xmax, ymax = map_coords(xmax, ymax, width_scl, height_scl)
    return img.crop((xmin, ymin, xmax, ymax))

def run_model_on_image(img, model, pattern_dir):
    # img is PIL
    # resize to 320
    img_size = 320
    original_width,original_height = img.size
    small_width,small_height = img_size, int(img_size/original_width*original_height)
    width_scl = original_width / small_width
    height_scl = original_height / small_height
    resized_image = img.resize((small_width,small_height))
    # loop over square crop of image and run the model
    i = 0
    log_coords = []
    while True:
        write_notif("scanning log page %d" % i, 25)
        # crop image into small and large versions
        cropped_small,end = crop_image_square(resized_image, i, small_width, small_height)
        cropped_large,end = crop_image_square(img, i, original_width, original_height)
        # if reached the end stop
        if not end:
            i += 1
        else:
            break
        write_notif("\nscan with model", 25, write=False)
        # get model res
        results = model(cropped_small)
        res = json.loads(results.pandas().xyxy[0].to_json())
        # process results 
        if res["name"]:
            j = 0
            for key, value in res["class"].items():
                if value == 2 or value == 0:
                    # pattern found
                    pattern = extract_image_from_res(cropped_large, res, key, width_scl, height_scl)
                    #cv2.imwrite(pattern_dir+"/%s_%d.jpg" % (key, i), pattern)
                    extract_patterns(np.array(pattern), pattern_dir+"/{}.jpg")
                if value == 1:
                    # log found
                    xmin = int(res["xmin"][key])
                    ymin = int(res["ymin"][key])
                    xmax = int(res["xmax"][key])
                    ymax = int(res["ymax"][key])
                    xmin, ymin = map_coords(xmin, ymin, width_scl, height_scl)
                    xmax, ymax = map_coords(xmax, ymax, width_scl, height_scl)
                    log_coords.append([xmin, ymin, xmax, ymax])

                j += 1
    return log_coords

"""
main script to run to get legend + patterns
"""
def process_pdf(pdf_path):
    # import model from trained weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='scripts/weights.pt')
    write_notif("model loaded", 5)

    # main path for saving images
    dir_path = os.path.join("data/images/",pdf_path[10:-4])
    pattern_dir = os.path.join(dir_path,"patterns")

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if not os.path.exists(pattern_dir):
        os.mkdir(pattern_dir)

    log_image = get_log_image(pdf_path, dpi=200, save=False)
    log_image = Image.fromarray(log_image)
    cv2.imwrite(dir_path+"/log.jpg", log_image)
    write_notif("log page found", 15)
    # extract log image
    #cv2.imwrite(dir_path+"/log.jpg", log_image)
    write_notif("log page found", 15)

    # run model, also finds and saves patterns
    # TODO could pass patterns in memory instead of writing and reading from disk
    # returns coordinates of log for each image
    log_coords = run_model_on_image(log_image, model, pattern_dir)
    if len(log_coords) == 0:
        write_notif("log not found", 35)
        write_notif("\ntrying alternative method", 35, write=False)
        log_path = find_log_legend(log_image, dir_path, pattern_dir)
    else:
        write_notif("straightening log", 40)
        log_path = separate_log(log_coords, log_image, dir_path+"/log_{}.jpg")

    resize_patterns(pattern_dir)
    write_notif("clustering log", 50)
    out = cluster_log(dir_path, pattern_dir)

    write_notif("ty placement", 90)
    #tf_out = finale(log_path)
    print("ty")

    write_notif("end", 100)
    return {"info": "finished :D",
            "data": out,
            "pdf": pdf_path[10:]}
