import os, sys
from enum import Enum
from typing import Union

from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI,Request,Path,File, UploadFile
from pydantic import BaseModel

# custom package
from scripts.pdftoimage import convert_path

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

@app.get("/pie/")
async def pie(
        request: Request,
        name : str = "Well",
        deepness1 : float = 0,
        deepness2 : float = 0,
        sandstone: float = 0,
        clay: float = 0,
        latitude: str = "Not available in this document",
        longitude: str = "Not available in this document",
        description: Union[str, None] = None
    ):

    return templates.TemplateResponse("pie.html", {
            "request": request,
            "name": name,
            "deepness1": deepness1,
            "deepness2": deepness2,
            "sandstone": sandstone,
            "clay":clay,
            "latitude": latitude,
            "longitude": longitude,
            "description": description
        })

@app.exception_handler(404)
async def http_exception_handler(request, exc):
    return RedirectResponse("/404")

@app.post("/upload-pdf")
async def create_upload_file(file: UploadFile = File(...)):
    if file.headers["content-type"] != "application/pdf":
        return {"file": file,
                "error": "not a pdf"}
    path = "data/pdfs/"+file.filename
    try: 
        contents = file.file.read()
        with open(path, "wb+") as f:
            f.write(contents)
    except Exception as e:
        return {"path": path, 
                "error": "Error uploading the file {}".format(e)}
    finally:
        try:
            image_path, pages, seconds, size = convert_path(path, file.filename[:-4])
        except Exception as e:
            return {"file": file,
                    "path": path, 
                    "error": "Error with the file {}".format(e)}
        finally: 
            file.file.close()
    return {"path": path,
            "file": file.filename,
            "images": image_path, 
            "pages-converted": pages,
            "seconds": seconds, 
            "pdf-size": size,
            "info": "Uploaded {0}MB in {1}s".format(size, seconds)}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
