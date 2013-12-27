#! /usr/bin/env python

import os
import sys

import neurolab as nl

from glob import glob


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise Exception('Script takse exactly one parameter '+\
                        '(path to face database)')
    db_path = sys.argv[1]

    for path, content, files in os.walk(db_path):
        print files

