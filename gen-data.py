"""
Generate Data
-------------

Generate given data (Images and Annotations) to `rec` format
"""

import os
import time
import argparse

import numpy as np
import xml.etree.ElementTree as ET

import altusi.config as cfg
from altusi.logger import Logger

LOG = Logger(__file__.split('.')[0])

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str,
            required=True,
            help='Path to data folder')
    parser.add_argument('--name', '-n', type=str,
            required=False, default='data',
            help='Name of REC data')
    parser.add_argument('--verbal', '-v',
            required=False, default=False, action='store_true',
            help='Whether the log is shown')
    args = parser.parse_args()

    return args


def load_label(anno_path):
    root = ET.parse(anno_path).getroot()
    image_fname = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    image_shape = (height, width)
    bboxes = []
    IDs = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in cfg.CLASSES:
            continue
        cls_id = cfg.INDEX_MAP[cls_name]

        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text) - 1
        ymin = int(xml_box.find('ymin').text) - 1
        xmax = int(xml_box.find('xmax').text) - 1
        ymax = int(xml_box.find('ymax').text) - 1
        bboxes.append((xmin, ymin, xmax, ymax))
        IDs.append(cls_id)

    return image_fname, image_shape, bboxes, IDs


def processDataFolder(folder_path):
    """Process data in a sub-folder"""
    annos_path = os.path.join(folder_path, cfg.ANNO_DIR)
    cnt_annos = len(os.listdir(annos_path))
    LOG.info('annos count: {}'.format(cnt_annos))

    image_fnames = []
    image_shapes = []
    bboxes_lst = []
    IDs_lst = []
    for anno_file in os.listdir(annos_path):
        anno_path = os.path.join(annos_path, anno_file) 

        image_fname, image_shape, bboxes, IDs = load_label(anno_path)

        image_fnames.append(image_fname)
        image_shapes.append(image_shape)
        bboxes_lst.append(np.array(bboxes))
        IDs_lst.append(np.array(IDs))

    return image_fnames, image_shapes, bboxes_lst, IDs_lst

def write_line(img_path, img_shape, bboxes, ids, idx):
    H, W = img_shape
    A, B, C, D = 4, 5, W, H

    labels = np.hstack((ids.reshape(-1, 1), bboxes)).astype('float')

    if W ==0 or H == 0:
        print(img_path)
    labels[:, (1, 3)] /= float(W)
    labels[:, (2, 4)] /= float(H)

    labels = labels.flatten().tolist()

    str_idx = [str(idx)]
    str_header = [str(_) for _ in [A, B, C, D]]
    str_labels = [str(_) for _ in labels]
    str_path = [img_path]

    line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'

    return line

def createLSTFile(image_fnames, image_shapes, bboxes_lst, IDs_lst, name):
    with open(name + '.lst', 'w') as fw:
        for i, image_fname in enumerate(image_fnames):
            line = write_line(image_fname, 
                              image_shapes[i],
                              bboxes_lst[i],
                              IDs_lst[i],
                              i)
            fw.write(line)


def app(data_path, data_name, verbal=True):
    image_fnames, image_shapes, bboxes_lst, IDs_lst = \
        processDataFolder(data_path)
    
    train_cnt = int(cfg.TRAIN_RATIO * len(image_fnames))
    val_cnt = len(image_fnames) - train_cnt

    createLSTFile(image_fnames[:train_cnt], image_shapes[:train_cnt], 
                  bboxes_lst[:train_cnt], IDs_lst[:train_cnt], 'train-'+data_name)
    createLSTFile(image_fnames[train_cnt:], image_shapes[train_cnt:], 
                  bboxes_lst[train_cnt:], IDs_lst[train_cnt:], 'val-'+data_name)


def main(args):
    app(args.data, args.name, args.verbal)


if __name__ == '__main__':
    LOG.info('Task: Generate REC data\n')

    args = getArgs()
    main(args)

    LOG.info('Process done')

