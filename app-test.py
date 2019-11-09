"""
Test Trained model on Images
----------------------------
"""


import os
import time
import argparse

import numpy as np
import cv2 as cv

import mxnet as mx
from mxnet import gluon
import gluoncv as gcv

import altusi.config as cfg
import altusi.visualizer as vis
from altusi import helper, imgproc
from altusi.logger import Logger

LOG = Logger(__file__.split('.')[0])


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str,
                        required=False,
                        help='Path to image folder to test')
    parser.add_argument('--image', '-i', type=str,
                        required=False,
                        help='Path to image to test')
    parser.add_argument('--video', '-v', type=str,
                        required=False,
                        help='Path to Video source')
    parser.add_argument('--name', '-n', type=str,
                        required=False, default='camera',
                        help='Name of Video')
    parser.add_argument('--show', '-s',
                        default=False, action='store_true',
                        help='Whether the output is visualized')
    parser.add_argument('--record', '-r', 
                        default=False, action='store_true',
                        help='Whether the output is saved or not')
    args = parser.parse_args()

    return args



def filter_bboxes(bboxes, scores, class_ids, thresh=0.5):
    ids = np.where(scores.asnumpy().reshape(-1) > thresh)[0]

    if len(ids):
        return bboxes[ids], scores[ids], class_ids[ids]
    else:
        return None, None, None


def ssd_predict(net, image, ctx, thresh=0.5):
    x, img = gcv.data.transforms.presets.ssd.transform_test(mx.nd.array(image), short=512)
    x = x.as_in_context(ctx)

    class_ids, scores, bboxes = net(x)
    if len(bboxes[0]) > 0:
        bboxes, scores, class_ids = filter_bboxes(bboxes[0], scores[0], class_ids[0], thresh)
        if bboxes is not None:
            classes = [net.classes[int(idx.asscalar())] for idx in class_ids]

    return class_ids, scores, bboxes, img


def rescale_bboxes(bboxes, dims, new_dims):
    H, W = dims
    _H, _W = new_dims

    _bboxes = []
    for bbox in bboxes:
        bbox = bbox.asnumpy()
        bbox = bbox / np.array([W, H, W, H]) * np.array([_W, _H, _W, _H])
        _bboxes.append(bbox)

    return _bboxes

def processImage(net, ctx, image):
    cls_ids, scores, bboxes, ssd_image = ssd_predict(net, image, ctx)

    if bboxes is not None:
        scores = scores.reshape(-1).asnumpy()
        bboxes = rescale_bboxes(bboxes, ssd_image.shape[:2], image.shape[:2])
        image = vis.plotBBoxes(image, bboxes, 
                               classes=len(bboxes)*['cow'], 
                               scores=scores)
        image = cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB)

    return image


def testImage(net, ctx, image_path):
    image = cv.imread(image_path)
    _, image_name = helper.getFilename(image_path)
    
    _start_t= time.time()
    image = processImage(net, ctx, image)
    _prx_t = time.time() - _start_t
    LOG.info('FPS: {:.4f}'.format(1/_prx_t))

    cv.imwrite('./saved-images/'+image_name, image)


def testFolder(net, ctx, folder_path):
    for i, image_file in enumerate(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, image_file)

        testImage(net, ctx, image_path)


def testVideo(net, ctx, video_link, video_name, show=False, record=False):
    cap = cv.VideoCapture(video_link)
    (W, H), FPS = imgproc.cameraCalibrate(cap)
    LOG.info('Camera Info: ({}, {}) - {:.3f}'.format(W, H, 30))

    if record:
        time_str = time.strftime(cfg.TIME_FM)
        writer = cv.VideoWriter(video_name+time_str+'.avi',
                                cv.VideoWriter_fourcc(*'XVID'), FPS, (W, H))

    while cap.isOpened():
        _, frm = cap.read()
        if not _:
            LOG.info('Reached the end of Video source')
            break

        _start_t = time.time()
        frm = processImage(net, ctx, frm)
        _prx_t = time.time() - _start_t
        LOG.info('FPS: {:.4f}'.format(1/_prx_t))

        if record:
            writer.write(frm)

        if show:
            cv.imshow(video_name, frm)
            key = cv.waitKey(1)
            if key in [27, ord('q')]:
                LOG.info('Interrupted by Users')
                break

    if record:
        writer.release()
    cap.release()
    cv.destroyAllWindows()


def main(args):
    BASE_MODEL = 'ssd_512_mobilenet1.0_custom'
    TRAINED_MODEL = 'ssd_512_mobilenet1.0_gym.params'

    BASE_MODEL = 'ssd_512_resnet50_v1_custom'
    TRAINED_MODEL = 'ssd_512_resnet50_v1_cow.params'

    net = gcv.model_zoo.get_model(BASE_MODEL, 
                                  classes=cfg.CLASSES, 
                                  pretrained_base=False)
    net.load_parameters(TRAINED_MODEL)
    ctx = mx.context.gpu(0) if mx.context.num_gpus() else mx.context.cpu()
    LOG.info('Device in Use: {}'.format(ctx))
    net.collect_params().reset_ctx(ctx)

    if args.data:
        testFolder(net, ctx, args.data)
    elif args.image:
        testImage(net, ctx, args.image)
    elif args.video:
        testVideo(net, ctx, args.video, args.name, args.show, args.record)
    else:
        LOG.error('Specify test option')


if __name__ == '__main__':
    LOG.info('Task: Test on Images\n')

    args = getArgs()
    main(args)

    LOG.info('Process done')
