#!/bin/sh

yes | sudo rm -rf data/images/*/*
yes | sudo rmdir data/images/*
yes | sudo rm data/json/*
yes | sudo rm data/pdfs/*
tree data/
