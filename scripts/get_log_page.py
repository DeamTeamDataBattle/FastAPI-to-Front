#!/bin/python3
import os, sys, cv2, numpy as np, time, glob, pypdfium2 as pdfium, matplotlib.pylab as plt, pytesseract, matplotlib, json
from PIL import Image
from scripts.functions import write_notif
"""
this file contains functions that take the whole pdf and 
finds the log page.
"""

DEBUG = False

IMAGE_WIDTH = 320

HEIGHT = IMAGE_WIDTH

def open_pdf(path):
    return pdfium.PdfDocument(path)

# returns the index of the longest page in the pdf
def get_longest_page_index(pdf):
    page_index = None
    max_height = 0
    for i,page in enumerate(pdf):
        h = page.get_height()
        if h > max_height:
            max_height = h
            page_index = i
    return page_index

# searches the pdf summary to find log page index
def find_log_page_in_summary(pdf):
    LOG_PAGE = None
    for item in pdf.get_toc():
        title = item.title.lower().split(" ")
        if "log" in title and ("composite" in title or "complete" in title or "completion" in title):
            LOG_PAGE = item.page_index
            break
    return LOG_PAGE

# search image for text
def search_image_text(image, key_words):
    FOUND = False
    for w in key_words:
        if w.lower() in text:
            FOUND = True
    return FOUND

# model that checks that it is the right page
def verify_log_page(img, debug=False):
    img = img[0:HEIGHT]
    text = pytesseract.image_to_string(img).lower().replace("\n", "").split(" ")
    if debug:
        print("Page %d\n%s" % (index, text))
    if "log" in text and ("composite" in text or "completion" in text):
        return True
    else:
        return False

# convert page to image
def convert_page_to_image(pdf, page_index, dpi=150, write=""):
    img = pdf[page_index].render(scale = dpi / 72).to_numpy()
    if write != "":
        cv2.imwrite(write+"log_page.jpg", img)
    return img

# check all the pages and find log
def find_log_page(pdf, dpi=150, debug=False):
    n_pages = len(pdf)
    page_indices = [n_pages-i-1 for i in range(n_pages)]  # all pages
    key_words = ["LOG", "COMPOSITE", "COMPLETION"]
    for index in page_indices:
        img = pdf[index].render(scale = dpi / 72).to_numpy()
        text = [''.join(filter(str.isalpha, s)) for s in pytesseract.image_to_string(img[0:HEIGHT]).lower().replace("\n", "").split(" ")]
        if debug:
            print("Page %d\n%s" % (index, text))
        if "log" in text and ("composite" in text or "completion" in text):
            return img, index
    else:
        return [], index

def split_and_save(save_path, image, img_size):
    img = Image.fromarray(image)
    w,h = img.size
    W,H = img_size, int(img_size/w*h)
    img = img.resize((W,H))
    head = img.crop((0, 0, W, img_size*4))
    for i in range(4):
        cropped = head.crop((0, i*img_size, W, (i+1)*img_size))
        cropped.save(save_path.format("head_%d" % (i+1)))
    img.save(save_path.format("log"))

def get_log_image(path, dpi=150, save=True):
    write_notif("scanning pdf for log page\n",10)
    save_path = os.path.join(os.path.curdir,"data/images/{0}_"+path[10:-4]+".jpg")
    pdf = open_pdf(path)
    if len(pdf) == 1:
        return convert_page_to_image(pdf, 0, dpi=dpi)
    log_page = find_log_page_in_summary(pdf)
    if not log_page:
        write_notif("not in summary checking longest\n", 11, write=False)
        log_page = get_longest_page_index(pdf)
        log_image = convert_page_to_image(pdf, log_page, dpi=dpi)
        if not verify_log_page(log_image):
            write_notif("not longest checking all\n", 12, write=False)
            log_image, log_page = find_log_page(pdf, dpi=150, debug=DEBUG)
            if len(log_image) == 0:
                write_notif("log page not found", 100)
                raise Exception("log not found")
    else:
        write_notif("found in summary\n",13, write=False)
    write_notif("found log page: %d" % log_page, 14, write=False)
    log_image = convert_page_to_image(pdf, log_page, dpi=dpi)
    if save:
        split_and_save(save_path, log_image, IMAGE_WIDTH)
    return log_image

# get all the pdf files and save the log page
if __name__ == "__main__":
    args = ['-f', '-d', '-I']
    indexes = [sys.argv.index(a) for a in args if a in sys.argv]
    if len(indexes) > 1:
        files = sys.argv[indexes[0]+1:indexes[1]]
    else:
        files = sys.argv[indexes[0]+1:]
    if '-d' in sys.argv:
        DEBUG = True
    else:
        DEBUG = False
    if '-I' in sys.argv:
        index = sys.argv.index('-I')
        IMAGE_WIDTH = int(sys.argv[index+1])
        print("image width: %d " % IMAGE_WIDTH)
        HEIGHT = IMAGE_WIDTH

    for path in files:
        try:
            get_log_image(path, dpi=150)
        except Exception as e:
            print("error", e)


