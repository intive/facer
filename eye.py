#! /usr/bin/env python

import argparse
import cv2
import numpy

from detectors import (ycrcb_skindetection, hsv_param_skindetection,
                       rgb_param_skindetection)


METHOD_MAPPER = {'hsv': hsv_param_skindetection,
                 'rgb': rgb_param_skindetection,
                 'ycrcb': ycrcb_skindetection,
                 }

# Main `function`
if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-m', '--method', type=str, default='ycrcb')
    arguments = input_parser.parse_args()

    method = arguments.method.lower()
    mks_iterator = METHOD_MAPPER.keys().index(method)

    cam_index = 0

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)

        mapper = METHOD_MAPPER[method](frame)

        cv2.imshow("preview", frame)
        frame_mapped = frame.copy()
        if len(mapper.shape) == 2:
            frame_mapped[:, :, 0] = frame[:, :, 0] * mapper
            frame_mapped[:, :, 1] = frame[:, :, 1] * mapper
            frame_mapped[:, :, 2] = frame[:, :, 2] * mapper
        else:
            frame_mapped = frame * mapper
        cv2.imshow("frame mapped", frame_mapped)

        # KEY binding section {{{
        key = cv2.waitKey(10)
        # key `q` and ESC
        if key in [27, 113]:  # exit on ESC
            break

        # `c`
        if key == 99:
            cam_index += 1
            vc = cv2.VideoCapture(cam_index%2)

        # key `m`
        if key == 109:
            cv2.destroyAllWindows()
            mks = METHOD_MAPPER.keys()
            mks_iterator += 1
            method = mks[mks_iterator % len(mks)]
        # KEY binding section }}}

    cv2.destroyWindow("preview")
