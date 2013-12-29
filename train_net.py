#! /usr/bin/env python

import cv2
import os
import sys

import neurolab as nl
import numpy as np

from glob import glob
from os import path


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        raise Exception('Script takse exactly one parameter '+\
                        '(path to face database)')
    db_path = sys.argv[1]
    nodb_path = sys.argv[2]
    rel_dir = path.dirname(path.abspath(__file__))
    image_w = 23
    image_h = 28

    #initialize nerual network
    input_size = image_w * image_h
    input_def = np.zeros((input_size, 2))
    input_def += [0, 1]
    net = nl.net.newff(input_def, [10, 5, 1])

    p_input = []
    n_input = []
    for path_d, content, files in os.walk(db_path):
        for ifile in files:
            if path.splitext(ifile)[1] in ['.pgm', '.png', '.jpg']:
                ifile = path.join(path_d, ifile)
                img = cv2.imread(ifile, 0)
                img.resize(image_w, image_h)
                img = img.reshape(img.size)
                p_input.append(img)
    for path_d, content, files in os.walk(nodb_path):
        for ifile in files:
            if path.splitext(ifile)[1] in ['.pgm', '.png', '.jpg']:
                ifile = path.join(path_d, ifile)
                img = cv2.imread(ifile, 0)
                img.resize(image_w, image_h)
                img = img.reshape(img.size)
                n_input.append(img)
    p_input = np.array(p_input)
    n_input = np.array(n_input)
    #input = input[:1]
    p_target = np.ones(len(p_input)).reshape((len(p_input),1))
    n_target = np.zeros(len(n_input)).reshape((len(n_input),1))
    #target = target[:1]

    net.train(p_input, p_target)
    net.train(n_input, n_target)
    net.save('face_neural.net')



