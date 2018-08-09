import tensorflow as tf
from os import path
import numpy as np
from scipy import misc
from styx_msgs.msg import TrafficLight
from distutils.version import LooseVersion

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
        if self.sess == None:
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(self.checkpoint, 'rb') as fid:
                    print(self.checkpoint)
                    serialized_graph = fid.read()
                    print("fid")
                    print(len(serialized_graph))
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                    
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # Get handles to input and output tensors
                for op in tf.get_default_graph().get_operations():
                    print(op.name)
        '''
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        print("names")
        print(all_tensor_names)
        tensor_dict = {}
        for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # Run inference
        output_dict = self.sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(img, 0)})
        '''     
        return TrafficLight.UNKNOWN