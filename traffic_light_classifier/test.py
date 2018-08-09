from site_model import SiteModel
import numpy as np
import cv2

image = cv2.imread('real_img/G_0.png')
model=SiteModel('models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb')
print("testing..")
result=model.predict(image)
print(result)