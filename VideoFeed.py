import numpy as np
import math
import os
import time
import multiprocessing
import tensorflow as tf
import argparse
import subprocess
import PIL.Image as Image
import shlex
import utils.glprocess as proc

from os.path import join
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from threading import Thread
from multiprocessing import Queue,cpu_count, Pool
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels

try:
    import cv2 as cv
    print('OpenCV version:', cv.__version__)
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" subdirectory if required)')
print('TensorFlow version:', tf.__version__)                      
if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

dir = os.getcwd()

# Module config
MODEL_PATH= os.path.join(dir,'data','new18k.pb')
LABELS_PATH = os.path.join(dir,'data','new18k.pbtxt')
VIDEO_PATH = os.path.join(dir,'test','data.avi')
MAX_RESULTS =8

#loading label map
label_map = label_map_util.load_labelmap(LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_RESULTS,use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# print(category_index)

def initialize_tensor():
    print('Loading tensor model...')    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            print('loading complete.')        
        sess = tf.Session(graph=detection_graph)
    return sess,detection_graph

def detect_objects(image_np, sess,detection_graph):    
    height, width = image_np.shape[:2]
    # print('height-> %s, width-> %s' %(height,width))
    # max_height = 640
    # max_width = 480

    # center = (height / 2, width / 2)
    # # flip the image 
    # image_np = cv.flip( image_np, 1 )

    # # only shrink if img is bigger than required
    # if max_height < height or max_width < width:
    #     # get scaling factor
    #     scaling_factor = max_height / float(height)
    #     if max_width/float(width) < scaling_factor:
    #         scaling_factor = max_width / float(width)

    #     # resize image
    #     image_np = cv.resize(image_np, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

    #crop image    
    crop_width = 50
    crop_height = 50
    image_np=cv.flip(image_np,flipCode=0)
    image_np = image_np[crop_height+20:height-150, crop_width+80:width-50]

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
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
    
    boxes =np.around(boxes,decimals=2)
    boxes =np.squeeze(boxes)
    
    scores = np.around(scores,decimals=2)
    scores = np.squeeze(scores)
    
    classes = np.squeeze(classes).astype(np.int32)

    packed = proc.pack_objs(image_np,category_index,boxes,classes,scores,MAX_RESULTS)
    proc.map_objs(packed)                    

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)
    # image_np = cv.resize(image_np,None,fx=1.5, fy=1.5, interpolation = cv.INTER_CUBIC)        
    return image_np

def worker(input_q, output_q):    
    sess, detection_graph = initialize_tensor()
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb,sess,detection_graph))

    fps.stop()
    sess.close()

if __name__ == "__main__":
    queue_size = 20                        
    count = 0
    width = 640
    height = 480
           
    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    pool = Pool(1, worker, (input_q, output_q))

    video_capture = cv.VideoCapture(VIDEO_PATH)
    video_capture.set(cv.CAP_PROP_FRAME_WIDTH, width)  
    video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)  
    fps = FPS().start()    

    print('[INFO] Starting video feed...')
    
    # subprocess.Popen('"C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe" --qt-minimal-view --no-qt-name-in-title -f --playlist-autostart --loop --playlist-tree "%s"'%LOOPER_VID,
    #     stderr=subprocess.STDOUT,
    #     shell=True) 
    while(video_capture.isOpened()):
        ret, frame = video_capture.read()
        if ret==True:
            if count%10 ==0:
                input_q.put(frame)
                t = time.time()
                output_rgb = cv.cvtColor(output_q.get(), cv.COLOR_RGB2BGR)
                cv.imshow('Video', output_rgb)
                fps.update()
            count=count+1

        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()    
    print('[INFO] Terminating Camera feed...')

    pool.terminate()
    video_capture.release()
    cv.destroyAllWindows()
    print('[INFO] Process Stopped.')