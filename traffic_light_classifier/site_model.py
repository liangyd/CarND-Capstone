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
        #assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
        #print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        #if not tf.test.gpu_device_name():
        #    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        #else:
        #    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

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
                 #print(self.checkpoint)
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
                
                #crop the image, we only analyze the upper half of the image for traffic light detection
                w,h=img.shape[:2]
                img=img[0:h//2, 0:w]
                #load image into numpy array
                im_pil = Image.fromarray(img)
                (im_width, im_height) = im_pil.size
                image_np= np.array(im_pil.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (pred_boxes, pred_scores, pred_classes) = sess.run( [detection_boxes, detection_scores, detection_classes], feed_dict={image_tensor: image_np_expanded})
                pred_boxes = pred_boxes.squeeze()
                pred_scores = pred_scores.squeeze() # the scores are in descreding order
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
                    # take the first one which has the largest confidence
                    if traffic_light is not None: break

                if traffic_light is None:
                    pass
                else:
                    tl_h, tl_w = traffic_light.shape[:2]
                    tl_h_side=int(tl_h*0.05)
                    tl_w_side=int(tl_w*0.3)
                    traffic_light=traffic_light[tl_h_side:tl_h-tl_h_side, tl_w_side:tl_w-tl_w_side] # crop the traffic light image, focus on the lights
                    brightness = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)[:,:,-1] 
                    hb, wb = np.where(brightness >= (brightness.max()-5))  # find the brightest area
                    hd, wd = np.where(brightness <= (brightness.min()+60))  # find the dark area
                    hb_mean = np.mean(hb)
                    hd_mean =np.mean(hd)
                    tl_h, tl_w = traffic_light.shape[:2]
                    pos_b = hb_mean/tl_h   # the position of the brightest area
                    pos_d =hd_mean/tl_h

                    if (pos_b < 0.45 and pos_d>0.5):
                        return TrafficLight.RED
                    elif (pos_b > 0.45 and pos_d<0.4):
                        return TrafficLight.GREEN
                    else:
                        return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN