from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import yaml
import time
import multiprocessing
import signal

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer

from threading import Thread
from multiprocessing import Queue, Pool

import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# constants
dir = os.getcwd()
weights = os.path.join(dir, 'model_final.pkl')  # path to model
model_cfg = os.path.join(dir, 'e2e_mask_rcnn_R-101-FPN_2x.yaml')  # yaml config file
video_path = os.path.join(dir, 'My.mp4')


def worker(input_q, output_q, model, dataset):
    try:
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = input_q.get()
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        output_im = vis_utils.vis_one_image_opencv(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            cls_boxes,
            cls_segms,
            cls_keyps,
            thresh=0.7,
            kp_thresh=2,
            show_box=False,
            dataset=dummy_coco_dataset,
            show_class=True
        )
        output_q.put(output_im)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()

def check_data():    
    if cfg is not None:
        assert os.path.exists(model_cfg)
    if video_path is not None:        
        assert os.path.exists(video_path)
    if weights is not None:        
        assert os.path.exists(weights)

def main():
    logger = logging.getLogger(__name__)
    logger.info('Loading config {} -> {}  Processing Video -> {}'.format(model_cfg, weights,video_path))
    merge_cfg_from_file(model_cfg)
    cfg.TEST.WEIGHTS = weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    num_workers = 1
    queue_size = 5

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    pool = Pool(num_workers, worker,
                (input_q, output_q, model, dummy_coco_dataset))

    video_capture = cv2.VideoCapture(video_path)

    while(video_capture.isOpened()):
        ret, frame = video_capture.read()
        if ret == True:
            input_q.put(frame)

        cv2.imshow('Video', output_q.get())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pool.close()
    video_capture.stop()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    check_data()
    main()
