from enum import Enum
from typing import Union

from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from fastapi import FastAPI,Request,Path
from pydantic import BaseModel


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
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/test/")
async def test(
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

    return templates.TemplateResponse("index.html", {
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