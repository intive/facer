#! /usr/bin/env python

import cv2
import os
import sys
import time

import neurolab as nl
import numpy as np

from glob import glob
from os import path
from pybrain.datasets import ClassificationDataSet
from pybrain.structure import FeedForwardNetwork, FullConnection, SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter


def normalize_image(img):
    img = img.astype(float)
    out = img - img.min()
    out /= out.max()
    img = img.astype(np.uint8)
    return out

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        raise Exception('Script takse exactly one parameter '+\
                        '(path to face database)')
    db_path = sys.argv[1]
    nodb_path = sys.argv[2]
    rel_dir = path.dirname(path.abspath(__file__))

    #initialize variables
    image_w = 23
    image_h = 28

    input_size = image_w * image_h

    #initialize nerual network
    net = FeedForwardNetwork()
    ## create layers
    in_layer = SigmoidLayer(image_w*image_h)
    hidden_layer1 = SigmoidLayer(30)
    hidden_layer2 = SigmoidLayer(15)
    out_layer = SigmoidLayer(1)
    ## add layers to network
    net.addInputModule(in_layer)
    net.addModule(hidden_layer1)
    net.addModule(hidden_layer2)
    net.addOutputModule(out_layer)
    ## add connections between layers
    net.addConnection(FullConnection(in_layer, hidden_layer1))
    net.addConnection(FullConnection(hidden_layer1, hidden_layer2))
    net.addConnection(FullConnection(hidden_layer2, out_layer))
    ## sorting modules topologically
    net.sortModules()
    #net = nl.net.newff(input_def, [40, 1])

    ds = ClassificationDataSet(image_w*image_h, class_labels=['face', 'noface'])

    p_input = []
    n_input = []
    for path_d, content, files in os.walk(db_path):
        for ifile in files:
            if path.splitext(ifile)[1] in ['.pgm', '.png', '.jpg']:
                ifile = path.join(path_d, ifile)
                img = cv2.imread(ifile, 0)
                img = cv2.resize(img, (image_w, image_h))
                img = img.reshape(img.size)
                img = img.astype(float)
                #img = normalize_image(img)
                ds.appendLinked(img.tolist(), [0])
                #ds.addSample(tuple(img), (1))
                #p_input.append(img)
    for path_d, content, files in os.walk(nodb_path):
        for ifile in files:
            if path.splitext(ifile)[1] in ['.pgm', '.png', '.jpg']:
                ifile = path.join(path_d, ifile)
                img = cv2.imread(ifile, 0)
                img = cv2.resize(img, (image_w, image_h))
                img = img.reshape(img.size)
                img = img.astype(float)
                #img = normalize_image(img)
                ds.appendLinked(img.tolist(), [1])
                #ds.addSample(tuple(img), (0))
                #n_input.append(img)
    #p_input = np.array(p_input)
    #n_input = np.array(n_input)
    #input = np.append(p_input, n_input, 0)
    #input = input[:1]
    #p_target = np.ones(len(p_input)).reshape((len(p_input),1))
    #n_target = np.zeros(len(n_input)).reshape((len(n_input),1))
    #target = np.append(p_target, n_target, 0)
    #target = target[:1]
    #print 'shape: ', input.shape
    #print 'target shape: ', target.shape
    #print 'target type: ', target.dtype
    #print 'min max input: ', input.min(), input.max()


    start = time.time()
    trainer = BackpropTrainer(net, ds)
    #ret = trainer.train()
    ret = trainer.trainUntilConvergence()
    #ret = net.train(input, target, goal=0.1)
    end = time.time()
    print 'Training error: ', ret
    print 'Time of training: ', end-start
    NetworkWriter.writeToFile(net, 'face_neural.xml')
    #net.save('face_neural.net')



