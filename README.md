## Real Time Multiple Object Tracking uisng YOLOv4, Tensorflow and DeepSORT
YOLOv4 (You Only Look Once version 4) is a real-time object detection algorithm. TensorFlow is an open-source machine learning framework, and DeepSort is a real-time multi-object tracking algorithm.You would use TensorFlow to implement the YOLOv4 model and run it on a video stream to detect the objects in real-time. Finally, you would use the DeepSort algorithm to track the detected objects over time by matching their detections across frames and estimating their positions.

## YOLOv4 Object Detection Model
Object detection using YOLOv4 model from scratch and have some basic concept over object detection model via the flow diagram how it works under the hood.

![This is an image](/images/AO.png)

YOLOv4 is a convolutional neural network (CNN) based object detection model. It uses a single neural network to predict bounding boxes and class probabilities directly from full images in one pass. The architecture of YOLOv4 consists of several layers, including:

1. A backbone network, which is responsible for extracting feature maps from the input image. In YOLOv4, the backbone network is a variant of the CSPDarknet architecture, which is a combination of the Darknet and Cross Stage Partial (CSP) architectures.

2. A neck network, which is used to fuse feature maps from the backbone network and extract higher-level features. In YOLOv4, the neck network consists of several SPP (Spatial Pyramid Pooling) and PAN (Path Aggregation Network) blocks.

3. A head network, which is used to predict bounding boxes and class probabilities from the features extracted by the neck network. The head network in YOLOv4 consists of several YOLO (You Only Look Once) blocks, which are similar to the YOLOv3 blocks but with some modifications.

4. A auxiliary network, which is used to enhance the feature maps and improve the accuracy of the prediction, The auxiliary network in YOLOv4 consists of SPADE (Spatially Adaptive Normalization) blocks and PAN blocks.

Overall, YOLOv4 architecture is more efficient and accurate than YOLOv3.

[Download file yolov4 model](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) and save it into YOLOv4_model folder.


## DeepSORT
![This is an image](/images/deepsort-architecture.jpg)

SORT performs very well in terms of tracking precision and accuracy. But SORT returns tracks with a high number of ID switches and fails in case of occlusion. SORT cannot handle occulsion that only relies on a simple motion model. What kind of occulsion occurs that SORT cannot tracked an object, i.e. if an object is obscured by another object that unable to tracked.

To overcome the problem of SORT, DeepSORT was introduced which tracks objects not only based on the velocity and motion of the object but also the appearence of the object. To train the deep association model in the DeepSORT cosine metric learning approach is used. Cosine Distance is a metric that helps the model recover identities in case of long term occlusion and motion estimation also fails.

#### Steps for Tracking an objects:
Step 1: Object detection and Recognition  

Step 2: Motion prediction and feature generation 

Step 3: Tracking 

## License
The MIT License (MIT). Please see [License File](/LICENSE) for more information.
