#! /usr/bin/env python

import argparse
import cv2
import numpy


def hsv_param_skindetection(img):
    proc_f = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mapr = proc_f[:, :, 0] >= 100
    mapr *= proc_f[:, :, 0] <= 255
    mapr *= proc_f[:, :, 1] >= 10
    mapr *= proc_f[:, :, 1] <= 200
    mapr *= proc_f[:, :, 2] >= 80
    mapr *= proc_f[:, :, 2] <= 180
    mapr *= img[:, :, 2] > img[:, :, 1]  # Rule R3 from Chiang Method
    mapr *= img[:, :, 1] > img[:, :, 0] - 20  # Rule R3 from Chiang Method
                                              # (modified `-20`)

    mapr = mapr * 1

    #create mapper for all chanels (from [x, y] to [x, y, 3])
    out = numpy.zeros((img.shape[0], img.shape[1], 3))
    for ln in range(out.shape[2]):
        out[:, :, ln] = mapr

    return out


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

    return out

METHOD_MAPPER = {'hsv': hsv_param_skindetection,
                 'rgb': rgb_param_skindetection,
                 }

# Main `function`
if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-m', '--method', type=str, default='hsv')
    arguments = input_parser.parse_args()
    if not arguments.method or \
       arguments.method.lower() not in ['hsv', 'rgb']:
        raise Exception('not allowed method')
    method = arguments.method
    mks_iterator = 0
    k = 0.01
    contur_img = False

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        mapper = METHOD_MAPPER[method](frame)

        frame *= mapper
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners = cv2.cornerHarris(gray, blockSize=5, ksize=7, k=k)

        #corners_idx = corners>frame[:, :, 1]
        #frame[:, :, 0][corners_idx] = corners[corners_idx]
        #frame[:, :, 1][corners_idx] = corners[corners_idx]
        #frame[:, :, 2][corners_idx] = corners[corners_idx]
        if contur_img:
            frame = corners

        frame = cv2.flip(frame, 1)

        key = cv2.waitKey(10)
        # key `q` and ESC
        if key in [27, 113]:  # exit on ESC
            break

        # key `m`
        if key == 109:
            mks = METHOD_MAPPER.keys()
            mks_iterator += 1
            method = mks[mks_iterator % len(mks)]

        # key `-` and `+`
        if key == 45:
            k -= 0.01
        if key == 61:
            k += 0.01

        # `k` binding
        if key == 107:
            contur_img = not contur_img

    cv2.destroyWindow("preview")
