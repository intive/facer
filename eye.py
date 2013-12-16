#! /usr/bin/env python

import cv2
import numpy

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


def hsv_param_skindetection(frame):
    proc_f = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mapr = proc_f[:,:,0]>=90
    mapr *= proc_f[:,:,0]<=200
    mapr *= proc_f[:,:,1]>=60
    mapr *= proc_f[:,:,1]<=200
    mapr *= proc_f[:,:,2]>=30
    mapr *= proc_f[:,:,2]<=255
    frame[:,:,0] = frame[:,:,0]*mapr
    frame[:,:,1] = frame[:,:,1]*mapr
    frame[:,:,2] = frame[:,:,2]*mapr
    return frame

def rgb_param_skindetection(frame):
    def create_r_mapper(r):
        up = -1.8423*(r*r) + 1.5294*r + 0.0422
        down = -0.7279*(r*r) + 0.6066*r + 0.1766
        return (up, down)

    proc_f = frame.astype(float)
    proc_f /= 255
    r = frame[:, :, 2] / (frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2])
    (up, down) = create_r_mapper(r)

    g = frame[:, :, 1] / (frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2])
    mapr = g<=up
    mapr *= mapr>=down

    b = numpy.arange(frame.shape[0] * frame.shape[1])\
             .reshape(frame.shape[0], frame.shape[1])
    b.fill(0)

    frame[:,:,0] = b.astype(numpy.uint8)
    frame[:,:,1] *= mapr
    frame[:,:,2] *= mapr

    return frame

while rval:
    cv2.imshow("preview", frame)

    rval, frame = vc.read()
    frame = hsv_param_skindetection(frame)

    key = cv2.waitKey(10)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")
