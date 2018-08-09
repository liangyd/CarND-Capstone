from styx_msgs.msg import TrafficLight
from sim_model import SimModel
from site_model import SiteModel

class TLClassifier(object):
    def __init__(self, is_site):
        # load classifier
        if (is_site):
            self.model=SiteModel("light_classification/models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb")
        else:
            self.model=SimModel()
        

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # implement light color prediction
        return self.model.predict(image)
