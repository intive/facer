#! /usr/bin/env python

import argparse
import cv2
import numpy


def calcHist(img):
    xs = numpy.unique(img)
    hist = numpy.histogram(img, bins=numpy.arange(257))  #compute img hist
    ind = hist[1][:-1]  #hist indexes
    hist = hist[0]  #hist values
    h = 201  # heigh of hist plot
    w = hist.size+1 # width of hist plot
    hist = hist.astype(float)
    hist = (hist/hist.max())*(h-1) # nomalize hist values to plot heigh
    hist = hist.astype(numpy.uint8)

    hist_shower = numpy.zeros((h, w))  # create hist plot image
    hist_shower = hist_shower.astype(numpy.uint8)
    hist_shower[(h-1)-hist, ind] = 255  # put white dots in hist values
    for i in ind:  # draw a bar
        hist_shower[(h-1)-hist[i]:, i] = 255

    cv2.imshow("mapper histogram", hist_shower)


def hsv_param_skindetection(img):
    proc_f = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mapr = proc_f[:, :, 0] >= 20
    mapr *= proc_f[:, :, 0] <= 255
    mapr *= proc_f[:, :, 1] >= 10
    mapr *= proc_f[:, :, 1] <= 255
    mapr *= proc_f[:, :, 2] >= 30
    mapr *= proc_f[:, :, 2] <= 255
    mapr *= img[:, :, 2] >= img[:, :, 1]  # Rule R3 from Chiang Method
    mapr *= img[:, :, 1] >= img[:, :, 0] - 20  # Rule R3 from Chiang Method
                                              # (modified `-20`)

    mapr = mapr * 1

    #create mapper for all chanels (from [x, y] to [x, y, 3])
    out = numpy.zeros((img.shape[0], img.shape[1], 3))
    for ln in range(out.shape[2]):
        out[:, :, ln] = mapr

    cv2.imshow('hsv map', out)

    return mapr

def rgb_param_skindetection(img):

    def create_r_mapper(r):
        up = -1.3767 * (r * r) + 1.0743 * r + 0.1452
        down = -0.776 * (r * r) + 0.5601 * r + 0.1766
        return (up, down)

    proc_f = img.astype(float)
    proc_f /= 255
    r = proc_f[:, :, 2] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    (up, down) = create_r_mapper(r)

    g = proc_f[:, :, 1] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    mapr = g <= up
    mapr *= mapr >= down

    b = numpy.arange(img.shape[0] * img.shape[1])\
             .reshape(img.shape[0], img.shape[1])
    b.fill(0)

    out = numpy.zeros((img.shape[0], img.shape[1], 3))
    out[:, :, 0] = mapr
    out[:, :, 1] = mapr
    out[:, :, 2] = mapr

    cv2.imshow('rgb map', out)
    return mapr

def ycrcb_skindetection(img):
    proc_f = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    proc_f = proc_f.astype(float)


    Y = proc_f[:,:,0]
    Cr = proc_f[:,:,1]
    Cb = proc_f[:,:,2]
    mapper_r = numpy.abs(Cr - 145)  # pixels close to 145 of Cr has highest importnace
    mapper_r = 255 - mapper_r
    mapper_r = (mapper_r - mapper_r.min())/(mapper_r.max() - mapper_r.min())
    mapper_r *= 255
    mapper_b = numpy.abs(Cb - 120)  # pixels close to 120 of Cb has highest importnace
    mapper_b = 255 - mapper_b
    mapper_b = (mapper_b - mapper_b.min())/(mapper_b.max() - mapper_b.min())
    mapper_b *= 255
    mapper = (mapper_r+mapper_b)/2  # mix Cr and Cb mapper in one channel
    mapper = mapper/255
    mapper = numpy.e**(-((mapper-1)**2)*10) #  Gaussian normalization
                                            #  puts higher weight to pixel
                                            #  closer to skin color candidates

    # pixels with similar values of Cr and Cb loose its importance
    short_indexes = Cr>(Cb-3)
    short_indexes *= Cr<(Cb+3)
    short_indexes *= Cb>(Cr+3)
    mapper[short_indexes] = 0.55-(mapper[short_indexes]-0.55)

    # normalize histogram to make face more clear
    mapper_normed = mapper * 255
    mapper_normed = mapper_normed.astype(numpy.uint8)
    mapper_normed = cv2.equalizeHist(mapper_normed)
    mapper = mapper_normed.astype(float)/255

    cv2.imshow("YCrCb face mapper", mapper_normed)
    return mapper


