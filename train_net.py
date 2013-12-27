#! /usr/bin/env python

import cv2
import os
import sys

import neurolab as nl
import numpy as np

from glob import glob
from os import path


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise Exception('Script takse exactly one parameter '+\
                        '(path to face database)')
    db_path = sys.argv[1]
    rel_dir = path.dirname(path.abspath(__file__))
    image_w = 23
    image_h = 28

    #initialize nerual network
    input_size = image_w * image_h
    input_def = np.zeros((input_size, 2))
    input_def += [0, 1]
    net = nl.net.newff(input_def, [10, 5, 1])

    input = []
    for path_d, content, files in os.walk(db_path):
        for ifile in files:
            if path.splitext(ifile)[1] in ['.pgm', '.png', '.jpg']:
                ifile = path.join(path_d, ifile)
                img = cv2.imread(ifile, 0)
                img.resize(image_w, image_h)
                img = img.reshape(img.size)
                input.append(img)
    input = np.array(input)
    #input = input[:1]
    target = np.ones(len(input)).reshape((len(input),1))
    #target = target[:1]

    net.train(input, target)



