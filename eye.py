#! /usr/bin/env python

import cv2
import numpy

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    proc_f = numpy.array(frame, dtype=numpy.float64)
    proc_f /= 256
    r = proc_f[:, :, 2] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    r *= 256
    r = r.round().astype(int)

    r.dtype = numpy.int
    g = proc_f[:, :, 1] / (proc_f[:, :, 0] + proc_f[:, :, 1] + proc_f[:, :, 2])
    g *= 256
    g = g.round().astype(int)
    g.dtype = numpy.int

    b = numpy.arange(frame.shape[0] * frame.shape[1])\
             .reshape(frame.shape[0], frame.shape[1])
    b.fill(0)

    frame[:, :, 2] = r
    frame[:, :, 1] = g
    frame[:, :, 0] = b

    key = cv2.waitKey(10)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
