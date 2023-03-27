import pypdfium2 as pdfium
import os, time

def create_images(pdf_path, pdf_name, image_dir):
    start = time.time()
    pdf = pdfium.PdfDocument(pdf_path)
    n_pages = len(pdf)
    page_indices = list(range(n_pages))
    renderer = pdf.render(
        pdfium.PdfBitmap.to_pil,
        page_indices = page_indices,
    )
    for i, image in zip(page_indices, renderer):
        image.save(image_dir+"/page_%d.jpg" % i)
    elapsed = round(time.time() - start, 3)
    return n_pages, elapsed

def convert_path(pdf_path, pdf_name):
    pages = 0
    seconds = 0
    IMAGE_DIR = os.path.join("data/images/", pdf_name)
    # check if dir exists and filled with images
    # if already converted don't
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)
        pages, seconds = create_images(pdf_path, pdf_name, IMAGE_DIR)
    else:
        # dir exists check if images exist
        # list file extensions in image dir
        files = [f[-3:] for f in os.listdir(IMAGE_DIR) if (os.path.isfile(os.path.join(IMAGE_DIR, f)) and len(f) > 4)]
        # no images in dir, create images
        if "jpg" not in files:
            pages, seconds = create_images(pdf_path, pdf_name, IMAGE_DIR)
    return IMAGE_DIR, pages, seconds, round(os.path.getsize(pdf_path)/1e6, 2)

