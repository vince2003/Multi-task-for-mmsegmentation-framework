import numpy as np

import cv2
import pdb
import json

def extract_polygon_region (image, polygons):
    N=2
    mask = np.zeros ((image.shape[0], image.shape[1]))
    for polygon in polygons:
        group_polygon=np.array([[polygon[n:n+N] for n in range(0, len(polygon), N)]])
        cv2.fillPoly( mask, group_polygon, 255 )
    return mask 

def take_location(path):
    polygons=[]
    with open(path) as json_file:
        data = json.load(json_file)
        #pdb.set_trace()
        for key, value in data.items():
            print (value['bbox'])
            polygons.append(value['bbox'])
    return polygons

def convert_mask(img_path, bobox_path):
    image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    polygons=take_location(bobox_path)
    mask_new=extract_polygon_region (image, polygons)
    cv2.imwrite('polygon_level.png', mask_new)


img_path='b00191.jpg'
bobox_path='b00191_anno.json'
convert_mask(img_path, bobox_path)

 
