import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
import argparse
import cv2

from os.path import join
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from threading import Thread
from multiprocessing import Queue, Pool
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels

dir = os.getcwd()

# tensor config
MODEL_PATH= os.path.join(dir,'data','frozen_inference_graph.pb')
LABELS_PATH = os.path.join(dir,'data','frozen_inference_graph.txt')
IMAGE_PATH = os.path.join(dir,'data','combined.jpg')
MAX_RESULTS = 3

#loading label map
label_map = label_map_util.load_labelmap(LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_RESULTS,use_display_name=True)
category_index = label_map_util.create_category_index(categories)

print(category_index)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        print('loading complete.')
    sess = tf.Session(graph=detection_graph)

def detect_objects(image_np):
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        
        sess.close()
        boxes =np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        
        newboxes = boxes[0:MAX_RESULTS]
        newscores = scores[0:MAX_RESULTS]
        newclasses = classes[0:MAX_RESULTS]    
        
        #print(newboxes,newscores,newclasses)
        
        for i in range(0,MAX_RESULTS):
            if(newscores[i]>.5):
                print(newboxes[i],newscores[i],category_index[newclasses[i]])
        
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)            
        return image_np

if __name__ == "__main__":

    img = cv2.imread(IMAGE_PATH,cv2.IMREAD_COLOR )
    # cv2.imshow('image',img)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dimage = detect_objects(frame_rgb)
    cv2.imshow('Video', dimage)
    while True: 
        if cv2.waitKey(1) & 0xFF == ord('q'):        
            cv2.destroyAllWindows()
    
    