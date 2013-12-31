#! /usr/bin/env python

import argparse
import cv2
import numpy

from detectors import (ycrcb_skindetection, hsv_param_skindetection,
                       rgb_param_skindetection, mean_shift_skindetecion,
                       viola_facedetector)

from datetime import datetime


METHOD_MAPPER = {'hsv': hsv_param_skindetection,
                 'rgb': rgb_param_skindetection,
                 'ycrcb': ycrcb_skindetection,
                 'meanshift': mean_shift_skindetecion,
                 'viola': viola_facedetector,
                 }

# Main `function`
if __name__ == '__main__':
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-m', '--method', type=str, default='meanshift')
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
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)

        mapper = METHOD_MAPPER[method](frame)
        # bring maper to <0;1>
        mapper = mapper.astype(float)
        mapper = (mapper - mapper.min())
        mapper /= mapper.max()

        frame_mapped = frame.copy()
        if len(mapper.shape) == 2:
            frame_mapped[:, :, 0] = frame[:, :, 0] * mapper
            frame_mapped[:, :, 1] = frame[:, :, 1] * mapper
            frame_mapped[:, :, 2] = frame[:, :, 2] * mapper
        else:
            frame_mapped = frame * mapper
        cv2.imshow("frame mapped", frame_mapped)

        # template matching {{{
        template = cv2.imread('face_template.png', 0)
        #template = cv2.resize(template, (160, 194))
        template = cv2.equalizeHist(template)
        mapper_normed = mapper*255
        mapper_normed = mapper_normed.astype(numpy.uint8)
        match_map = cv2.matchTemplate(mapper_normed, template,
                                      cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_map)
        top_left = min_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
        cv2.rectangle(frame, top_left, bottom_right, 255, 2)

        cv2.imshow("map match", match_map)

        #frame[:template.shape[0],:template.shape[1],0] = template
        #frame[:template.shape[0],:template.shape[1],1] = template
        #frame[:template.shape[0],:template.shape[1],2] = template
        # template matching }}}

        # KEY binding section {{{
        key = cv2.waitKey(10)
        # key `q` and ESC
        if key in [27, 113]:  # exit on ESC
            break

        # `p`
        if key == 112:
            cv2.imwrite('Screen_%s.png' % datetime.now().isoformat(), frame)
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
