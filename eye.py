#! /usr/bin/env python

import cv2
import numpy

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

def create_r_mapper(r):
    up = -1.3767*(r*r) + 1.0743*r + 0.1452
    down = -0.776*(r*r) + 0.5601*r + 0.1766
    return (up, down)

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    proc_f = numpy.array(frame, dtype=numpy.float64)
    proc_f /= 256
    r = proc_f[:, :, 2] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    (up, down) = create_r_mapper(r)


    g = proc_f[:, :, 1] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    mapr = g<=up
    mapr *= mapr>=down

    b = numpy.arange(frame.shape[0] * frame.shape[1])\
             .reshape(frame.shape[0], frame.shape[1])
    b.fill(0)

    frame[:, :, 2] *= mapr
    frame[:, :, 1] *= mapr
    frame[:, :, 0] *= mapr

    key = cv2.waitKey(10)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
