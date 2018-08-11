This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. 
It is an individual submission. All the tasks are completed by Yiding Liang(liangyd@umich.edu).

### System Description

## Perception
This subsystem determines the state of upcoming traffic lights and publishes their status (RED/YELLOW/GREEN/UNKNOWN).

# Traffic Light Detection (Site)
I used the Tensorflow object detection API to detect the traffic light in an image. I choose the pretrained ssd_mobilenet_v1_coco model for the traffic light detection. Since the tensorflow version in the Udacity's autonomous car is 1.3.0,  I need to use the export_inference_graph.py script to export the frozen inference graph from the checkpoint files. The outputs from this model are the bounding boxes, classes and scores. I select the bounding box with the highest confidence scores in the traffic light class. Then, I convert the BGR image to HSV to get the brightness of the each pixel. From the raw coloar image, it is difficult to classify a traffic light based on its color. Thus, I calculate the brightest area and darkest area which indicate the location of the light bulbs. The position of the bright and dark light bulbs determine whether the traffic light is green, yellow or red. 

To improve the detection performance, I only process the upper half of the image because the traffic light will not exist on the lower half. Also, I cropped the traffic light image to remove the influence of its bright edges. I tuned the parameters for the traffic light classifier.

# Traffic Light Detection (Simulator)
I used basic computer vision technique to calculate the number of red and green pixels in the image. If the area is larger than a threshold, the traffic light will be classified as red/green/yellow. 

## Planning 
The planning subsystem plans the vehicle’s path based on the vehicle’s current position and velocity along with the state of upcoming traffic lights. A list of waypoints to follow is passed on to the control subsystem

# Waypoint Updater
The waypoint updater publishes a list of final waypoints based on vehicle's current position, traffic and the base waypoints. It finds the closest waypoint to the vehicle's current position, and generates a list of future waypoints in the same direction as the vehicle moves. If the red traffic light is detected, it will generate another list of waypoints to decelerate the car before the stopline. 

## Control 
This subsystem publishes control commands for the vehicle’s steering, throttle, and brakes based on a list of waypoints to follow.

# Twist Controller
The throttle controller is a simple PID controller that compares the current velocity with the target velocity and adjusts the throttle accordingly. The throttle gains were tuned using trial and error for allowing reasonable acceleration without oscillation around the set-point. The yaw controller controller translates the proposed linear and angular velocities into a steering angle based on the vehicle’s steering ratio and wheelbase length. The brake is controlled to be proportional to the deceleration rate. 


