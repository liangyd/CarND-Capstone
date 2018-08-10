import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight
from distutils.version import LooseVersion
from PIL import Image
import cv2

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
        self.prob_thr = 0.07
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
                (pred_boxes, pred_scores, pred_classes) = sess.run( [detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np_expanded})
                pred_boxes = pred_boxes.squeeze()
                pred_scores = pred_scores.squeeze() # in descreding order
                pred_classes = pred_classes.squeeze()

                traffic_light = None
                h, w = img.shape[:2]
                for i in range(pred_boxes.shape[0]):
                    box = pred_boxes[i]
                    score = pred_scores[i]

                    if score < self.prob_thr: continue
                    if pred_classes[i] != self.TRAFFIC_LIGHT_CLASS: continue
                    x0, y0 = box[1] * w, box[0] * h
                    x1, y1 = box[3] * w, box[2] * h
                    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
                    area = np.abs((x1-x0) * (y1-y0)) / (w*h)
                    traffic_light = img[y0:y1, x0:x1]
                    # take the first one - with the most confidence
                    if traffic_light is not None: break

                if traffic_light is None:
                    pass
                else:
                    brightness = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2HSV)[:,:,-1] 
                    hs, ws = np.where(brightness >= (brightness.max()-30))
                    hs_mean = hs.mean()
                    print(hs_mean)
                    tl_h = traffic_light.shape[0]
                    print(tl_h)
                    if hs_mean / tl_h < 0.4:
                        return TrafficLight.RED
                    elif hs_mean / tl_h >= 0.55:
                        return TrafficLight.GREEN
                    else:
                        return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN