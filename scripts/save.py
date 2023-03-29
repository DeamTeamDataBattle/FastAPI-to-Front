import os, sys, json

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

