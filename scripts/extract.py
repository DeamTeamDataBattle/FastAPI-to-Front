from sklearn.cluster import KMeans
import cv2,os,json, sys, time
import numpy as np
import pandas as pd
from PIL import Image
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import disable_interactive_logging
from scripts.functions import LOG_COLUMN_SAVE_HEIGHT, LOG_COLUMN_SAVE_WIDTH, write_notif
disable_interactive_logging()

start = time.time()

# can't be less than 32
# 4 = good results
# 1.6 lowest
# 2 quick
scl = 3

SHAPE = [int(LOG_COLUMN_SAVE_HEIGHT*scl), int(LOG_COLUMN_SAVE_WIDTH*scl), 3]

base_model = VGG16(input_shape=SHAPE, weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract(img):
    img = cv2.resize(img, [SHAPE[1], SHAPE[0]])
    #img = img.convert('RGB') # Convert the image color space
    #x = image.img_to_array(img) # Reformat the image
    x = img
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)[0]# Extract Features
    feature = feature.reshape(-1,)
    return feature / np.linalg.norm(feature)

def get_features(imgs):
    feature = extract(imgs[0])
    features = np.zeros((len(imgs), feature.size))
    for i in range(len(imgs)):
        features[i] = extract(imgs[i])
    return features

def resize_image_width(img, width, height=None):
    h,w,d = img.shape
    if not height:
        height = int(width/w*h)
    return cv2.resize(img, (width, height))

def cluster_log(dir_path, pattern_dir) :
    log_file = dir_path+"/log_image.jpg"
    patterns = [f for f in os.listdir(pattern_dir) if "legend" in f]
    log_img = resize_image_width(cv2.imread(log_file), LOG_COLUMN_SAVE_WIDTH)
    #cv2.imwrite(dir_path+"/resize.jpg", log_img)
    log_h, log_w, d = log_img.shape
    print(patterns)
    pattern_imgs_orig = [resize_image_width(cv2.imread(os.path.join(pattern_dir, f)), LOG_COLUMN_SAVE_WIDTH) for f in patterns]
    pattern_height = max(i.shape[0] for i in pattern_imgs_orig)
    pattern_imgs = [resize_image_width(cv2.imread(os.path.join(pattern_dir, f)), LOG_COLUMN_SAVE_WIDTH, height=pattern_height) for f in patterns]

    # extract features
    features = get_features(pattern_imgs)

    # split log into sections
    N = log_h//pattern_height
    reconstructed = pattern_imgs[0]
    comp = []
    for i in range(1, N-1):
        write_notif("clustering\n%d/%d" % (i, (N-1)), percent=50+int(40*i/(N-1)))
        #print("%d/%d" % (i, (N-1)))
        query = extract(log_img[i*pattern_height:(i+1)*pattern_height])
        dists = np.linalg.norm(features - query, axis=1)
        dists = dists/np.sum(dists)
        ids = np.argsort(dists)[:2]
        dists = np.take_along_axis(dists, ids, axis=0)
        dists = dists/np.sum(dists)
        #img = cv2.addWeighted(pattern_imgs[ids[0]], dists[0], pattern_imgs[ids[1]], dists[1], 0)
        img = pattern_imgs_orig[ids[0]]
        reconstructed = cv2.vconcat((reconstructed, img))
        comp.append(ids[0])
    cv2.imwrite(dir_path+"/cluster.jpg", reconstructed)

    res = {'total':len(comp)}
    for i in range(len(patterns)):
        res[patterns[i][7:-4]] = comp.count(i)
    return res


if __name__ == "__main__":
    if "-f" not in sys.argv and len(sys.argv) > 2:
        raise Exception("file not given")
    else:
        path = sys.argv[sys.argv.index('-f')+1]
        dir_path = os.path.join("data/images",path[:-4])
        pattern_dir = os.path.join(dir_path, "patterns/")

        main(".", ".")

