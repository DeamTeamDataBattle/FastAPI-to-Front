import os, sys, time
from enum import Enum
from typing import Union

from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI,Request,Path,File, UploadFile
from pydantic import BaseModel

# custom package
from scripts.script import process
from scripts.save import save_results, check_if_already_processed, get_res, mv_images, get_patterns

# class Item(BaseModel):
#     name : str
#     deepness1 : float
#     deepness2 : float
#     sandstone : float
#     clay : float
#     latitude : str
#     longitude : str
#     description : Union[str, None] = None

app = FastAPI()
templates = Jinja2Templates(directory="static")

@app.exception_handler(404)
async def http_exception_handler(request, exc):
    return RedirectResponse("/404")

@app.get("/pie")
async def pie(
        request: Request,
        pdf: str = ""
    ):
    if pdf == "":
        # pdf not set redirecting
        return RedirectResponse(url="/")
    pdf_path = "data/pdfs/" + pdf
    if not check_if_already_processed(pdf_path):
        # hasn't run
        return RedirectResponse(url="/")

    print("copying images")
    mv_images(pdf_path)
    data = get_res(pdf_path)
    
    keys = ["sand", "clay", "lime", "shale", "salt", "silt", "chert"]
    values = [0 for i in range(len(keys))]
    comp = data["data"]
    total = 0
    # iterate over keys and compute values
    for key in comp:
        for i in range(len(keys)):
            if keys[i] in key:
                values[i] += comp[key]
                total += comp[key]


    return templates.TemplateResponse("pie.html", {
            "request": request,
            "name": data["pdf"] if "pdf" in data else "null",
            "deepness1": data["ty1"] if "ty1" in data else 0,
            "deepness2": data["ty2"] if "ty2" in data else 0,
            "sand": round(100*values[0]/total, 2),
            "clay": round(100*values[1]/total, 2),
            "latitude": data["lat"] if "lat" in data else "null",
            "longitude": data["lon"] if "lon" in data else "null", "description": data["desc"] if "desc" in data else "null",
        })

@app.get("/notif")
def get_notif():
    return {"notif": open("data/notification.txt", 'r').read()}

@app.get("/get-patterns")
def fetch_patterns():
    return get_patterns()

@app.post("/upload-pdf")
def create_upload_file(file: UploadFile = File(...)):
    start = time.time()
    if file.headers["content-type"] != "application/pdf":
        return {"file": file,
                "error": "not a pdf"}
    path = "data/pdfs/"+file.filename
    if not check_if_already_processed(path):
        print("new file, processing")
        try: 
            contents = file.file.read()
            with open(path, "wb+") as f:
                f.write(contents)
        except Exception as e:
            return {"path": path, 
                    "error": "Error uploading the file {}".format(e)}
        finally:
            try:
                res = process(path)
                save_results(path, res)
            except Exception as e:
                return {"file": file,
                        "path": path, 
                        "error": "Error with the file: {}".format(e)}
            finally: 
                file.file.close()
    else:
        print("already seen, fetching data")
        res = get_res(path)
    elapsed = round(time.time() - start, 3)
    return {"path": path,
            "file": file.filename,
            "info": "took: %.2fs" % elapsed} | res

app.mount("/", StaticFiles(directory="static", html=True), name="static")
