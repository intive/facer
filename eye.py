#! /usr/bin/env python
"""Face detector [eye.py].

Is using viola_jones modified Haar LVM method
in cascade vector training

"""

__author__ = 'Wiktor [wrutek] Rutka'
__email__ = 'wiktor.rutka@blstream.com'
__version__ = '0.1_predev'


import argparse
import cv2
import numpy as np

from detectors import (hsv_param_skindetection,
                       viola_facedetector)

from datetime import datetime


METHOD_MAPPER = {
    'viola': viola_facedetector,
}

# Main `function`
if __name__ == '__main__':
    input_parser = argparse.ArgumentParser(
        'Face detector.' +
        '\n\tInteractive commands:' +
        '\n\t `m` - change method' +
        '\n\t `p` - print screen' +
        '\n\t `s` - show with skin mapping' +
        '\n\t `c` - change cammera' +
        '\n\t `q/ESC` - quit' +
        '\n'
    )
    input_parser.add_argument('-m', '--method', type=str, default='viola',
                              help='current method is only viola: [viola]')
    arguments = input_parser.parse_args()

    method = arguments.method.lower()
    mks_iterator = METHOD_MAPPER.keys().index(method)
    faces = np.array([])

    cam_index = 0
    s_preview = False

    cv2.namedWindow("origin")
    vc = cv2.VideoCapture(cam_index)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("origin", frame)
        rval, frame = vc.read()
        frame = cv2.flip(frame, 1)
        preview = np.zeros(frame.shape)

        faces = METHOD_MAPPER[method](frame, faces)
        for i, (x, y, w, h) in enumerate(faces):
            mapr = hsv_param_skindetection(frame[y:y + h, x:x + w])
            preview[y:y + h, x:x + w] = \
                    frame[y:y + h, x:x + w].astype(float) * mapr
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (255, 0, 0), 2)
            cv2.putText(frame, 'Face: %d' % (i+1), (x, y), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 0 ,255))

        if s_preview:
            preview = preview.astype(np.uint8)
            cv2.imshow('preview', preview)
        # KEY binding section {{{
        key = cv2.waitKey(10)
        # key `q` and ESC
        if key in [27, 113]:  # exit on ESC
            break

        # `s` -> show preview
        if key == 115:
            s_preview = not s_preview
        # `p`
        if key == 112:
            cv2.imwrite('Screen_%s.png' % datetime.now().isoformat(), frame)
        # `c`
        if key == 99:
            cam_index += 1
            vc = cv2.VideoCapture(cam_index % 2)

        # key `m`
        if key == 109:
            cv2.destroyAllWindows()
            mks = METHOD_MAPPER.keys()
            mks_iterator += 1
            method = mks[mks_iterator % len(mks)]
        # KEY binding section }}}

    cv2.destroyWindow("preview")
