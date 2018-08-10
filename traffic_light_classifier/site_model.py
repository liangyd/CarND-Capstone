import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight
from distutils.version import LooseVersion
from PIL import Image

class SiteModel(object):
    def __init__(self, model_checkpoint):
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        self.sess = None
        self.checkpoint = model_checkpoint
        self.prob_thr = 0.15
        self.TRAFFIC_LIGHT_CLASS = 10
        tf.reset_default_graph()
        

    def predict(self, img):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.checkpoint, 'rb') as fid:
                 print(self.checkpoint)
                 serialized_graph = fid.read()
                 od_graph_def.ParseFromString(serialized_graph)
                 tf.import_graph_def(od_graph_def, name='')
                    
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                 # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                
                #load image into numpy array
                
                im_pil = Image.fromarray(img)
                (im_width, im_height) = im_pil.size
                image_np= np.array(im_pil.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes) = sess.run( [detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np_expanded})
                
        return TrafficLight.UNKNOWN