import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight


import tensorflow as tf

class SiteModel(object):
    def __init__(self, model_checkpoint):
        self.sess = None
        self.checkpoint = model_checkpoint
        self.prob_thr = 0.15
        self.TRAFFIC_LIGHT_CLASS = 10
        tf.reset_default_graph()
    
    def predict(self, img):
        return TrafficLight.UNKNOWN