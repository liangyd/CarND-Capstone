from site_model import SiteModel
import numpy as np
import cv2

image = cv2.imread('real_img/G_1.png')
model=SiteModel('remodel/frozen_inference_graph.pb')
print("testing..")
result=model.predict(image)
print(result)