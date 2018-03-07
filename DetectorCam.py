import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
import argparse

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
MODEL_PATH= os.path.join(dir,'data','frozen_inference_graph.pb')
LABELS_PATH = os.path.join(dir,'data','frozen_inference_graph.txt')
VIDEO_PATH = os.path.join(dir,'data','sipper.mp4')
MAX_RESULTS = 1

#loading label map
label_map = label_map_util.load_labelmap(LABELS_PATH)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_RESULTS,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)
# tracking_objects = ["swarovskispeaker","bottle","multiutilitykit","usbcharger","pebblebluetoothspeaker",
# "shantanukeychain","coffeemugblue","tabletop","passportholder"]
tracking_objects = ["bottle"]

isDefaultSet = False

default_location = dict()
object_tracking_dict = dict()

for obj in tracking_objects:
    default_location[obj]=dict([('xmin',0),('ymin',0),('xmax',0),('ymax',0),('isdefault',0)])

for obj in tracking_objects:
    object_tracking_dict[obj]=dict([('xmin',0),('ymin',0),('xmax',0),('ymax',0),('isMoved',0),('timer')])

def init_default(classes,boxes,scores):
    for i in range(MAX_RESULTS):
        if(newscores[i]>.5):
            classname = category_index[classes[i]]['name'] 
            currentObject = default_location[classname]
            if (currentObject['isdefault'] == 0):
                xmin,ymin,xmax,ymax = tuple(boxes[i].tolist())   
                currentObject[xmin]
                currentObject[ymin]
                currentObject[xmax]
                currentObject[ymax]
                print(classname,currentObject[xmin],
                currentObject[ymin],
                currentObject[xmax],
                currentObject[ymax])

def check_objects(classes,boxes):
    for i in range(MAX_RESULTS):        
        xmin,ymin,xmax,ymax = tuple(boxes[i].tolist())                                        
        if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']
        else:
            class_name = 'N/A'
        #get current state
        currentState = object_tracking_dict[class_name]
        xmin = currentState['xmin']
        ymin = currentState['ymin']
        xmax = currentState['xmax']
        ymax = currentState['ymax']

def detect_objects(image_np, sess, detection_graph):
    # height, width = image_np.shape[:2]
    # max_height = 300
    # max_width = 300

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

    boxes =np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes).astype(np.int32)
    
    newboxes = boxes[0:MAX_RESULTS]
    newscores = scores[0:MAX_RESULTS]
    newclasses = classes[0:MAX_RESULTS]    
   
    init_default(newclasses,newboxes,newscores)

    for i in range(0,MAX_RESULTS):
        if(newscores[i]>.5):            
            # check_objects(newclasses, newboxes)
            print(newboxes[i],newscores[i],category_index[newclasses[i]])
    
    # send for evaluation
    # check_objects(classes,boxes,scores)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)
    image_np = cv.resize(image_np,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)        
    return image_np

def worker(input_q, output_q):
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

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=500, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=500, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')                        
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))
    #video_capture = WebcamVideoStream(src=args.source,
    #                                  width=args.width,
    #                                  height=args.height).start()    

    video_capture = cv.VideoCapture(VIDEO_PATH)
    fps = FPS().start()

    while(video_capture.isOpened()):  # fps._numFrames < 120
        ret, frame =  video_capture.read()
        if ret==True:
            input_q.put(frame)

            t = time.time()

            output_rgb = cv.cvtColor(output_q.get(), cv.COLOR_RGB2BGR)
            cv.imshow('Video', output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv.destroyAllWindows()