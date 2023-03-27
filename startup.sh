#!/bin/sh

#sudo uvicorn main:app --reload --reload-dir static --reload-include *.js --reload-include *.html --port 80 --host 192.168.1.152
sudo uvicorn main:app --reload --port 80 --host 192.168.1.152
