#! /usr/bin/env python

from pybrain.tools.xml.networkreader import NetworkReader

import cv2

net = NetworkReader.readFrom('face_neural.xml')

img_p = cv2.imread('face_db/s1/1.pgm', 0)
img_n = cv2.imread('noface_db/3.png', 0)

img_p = cv2.resize(img_p, (23, 28))
img_n = cv2.resize(img_n, (23, 28))

img_p = img_p.reshape(23*28)
img_n = img_n.reshape(23*28)

ret_p = net.activate(img_p)
ret_n = net.activate(img_n)


print 'Result positive: ', ret_p
print 'Result negative: ', ret_n
