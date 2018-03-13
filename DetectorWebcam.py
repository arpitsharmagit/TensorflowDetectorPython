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
from multiprocessing import Queue, Pool
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

# tensor config
MODEL_PATH= os.path.join(dir,'data','new18k.pb')
LABELS_PATH = os.path.join(dir,'data','new18k.pbtxt')

LOOPER_VID = os.path.join(dir,'loopervids')
VLC = "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe -f --playlist-autostart --loop --playlist-tree %s"%LOOPER_VID
MAX_RESULTS =8

#loading label map
label_map = label_map_util.load_labelmap(LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_RESULTS,use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# print(category_index)

print('Loading tensor model...')    
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        print('loading complete.')
    global sess
    sess = tf.Session(graph=detection_graph)

def detect_objects(image_np, sess, detection_graph):   
    height, width = image_np.shape[:2]

    # flip the image 
    image_np = cv.flip( image_np, 1 )

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

    # print(boxes,classes)

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
    return image_np

def worker(input_q, output_q):    
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=20, help='Size of the queue.')                        
    args = parser.parse_args()
    
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()      
    fps = FPS().start()
    subprocess.Popen('"C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe" --qt-minimal-view --no-qt-name-in-title -f --playlist-autostart --loop --playlist-tree "%s"'%LOOPER_VID,
        stderr=subprocess.STDOUT,
        shell=True)
    print('[INFO] Starting camera feed...')
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        # print("frame shape=>", frame.shape)
        
        h,w = frame.shape[:2]
        x = 100
        y = 50
        frame = frame[y+20:h-150, x+10:w-50]         
        input_q.put(frame)

        t = time.time()

        output_rgb = cv.cvtColor(output_q.get(), cv.COLOR_RGB2BGR)
        cv.imshow('Video', output_rgb)
        # cv.imshow('Video', frame)
        fps.update()
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] Terminating Camera feed...')

    pool.terminate()
    video_capture.stop()
    cv.destroyAllWindows()
    print('[INFO] Process Stopped.')