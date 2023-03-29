import os, sys, json, shutil

def check_if_already_processed(pdf_path):
    json_file = os.path.join("data/json/",pdf_path[10:-4]+".json")
    return os.path.exists(json_file)

def get_res(pdf_path):
    json_file = os.path.join("data/json/",pdf_path[10:-4]+".json")
    with open(json_file, 'r') as f:
        return json.load(f)

def save_results(pdf_path, res):
    json_file = os.path.join("data/json/",pdf_path[10:-4]+".json")
    print(res)
    with open(json_file, "w+") as f:
        json.dump(res, f)

def mv_images(pdf_path):
    image_dir = os.path.join("data/images/",pdf_path[10:-4])
    pattern_dir = os.path.join(image_dir, "patterns")
    assets_dir = "static/images"
    files = [assets_dir+"/patterns/"+f for f in os.listdir(assets_dir+"/patterns/")]
    for f in files:
        os.remove(f)
    # copy log image
    shutil.copyfile(image_dir+"/log_image.jpg", assets_dir+"/log.jpg")
    shutil.copyfile(image_dir+"/cluster.png", assets_dir+"/cluster.jpg")
    files = [f for f in os.listdir(pattern_dir)]
    for f in files:
        shutil.copyfile(os.path.join(pattern_dir, f), assets_dir+"/patterns/"+f)

def get_patterns():
    files = ["/images/patterns/"+f for f in os.listdir("static/images/patterns/")]
    return {"files":files}
