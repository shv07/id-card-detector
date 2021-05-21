
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util, image_reader_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# Grab path to current working directory
#CWD_PATH = os.getcwd()
CWD_PATH = os.path.dirname(os.path.abspath(__file__))

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')


# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

def DetectIdentityCard(image_b64:str):
    """
    
    """
    if image_b64!=None and len(image_b64)>0:

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        # get the image in open cv format

        image = image_reader_util.base64_to_cv2_img(image_b64)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        '''
        image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=0.60)
        '''

        
        shape = np.shape(image)
        im_width, im_height = shape[1], shape[0]
        
        boxes1, scores1 = np.squeeze(boxes), np.squeeze(scores)

        result_boxes = [tuple([tuple(boxes1[i].tolist()), scores1[i]]) for i in range(boxes1.shape[0]) if scores1[i]>0.60]

        result_boxes1 = []

        for box, score in result_boxes:
            ymin, xmin, ymax, xmax = box
            ymin, xmin, ymax, xmax = int(ymin* im_height), int(xmin * im_width), int(ymax* im_height), int(xmax * im_width)
            result_boxes1.append(tuple([tuple([xmin, ymin, xmax-xmin, ymax-ymin]), score]))

        return result_boxes1
    return None

def DetectIdentityCardsCount(imageB64:str)->int:
    if imageB64==None or imageB64=="":
        return 0
    boxes = DetectIdentityCard(imageB64)
    return 0 if boxes==None else len(boxes)

#test
if __name__=="__main__":
    import base64

    testImg = './static/cell-phones.png'

    with open(testImg, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    boxes = DetectIdentityCard(encoded_string)

    image = cv2.imread(testImg)

    for box,score in boxes:
        x,y,w,h = box
        image = cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 5)

    cv2.imshow("Result", image)

    # Press any key to close the image
    cv2.waitKey(0)

    # Clean up
    cv2.destroyAllWindows()