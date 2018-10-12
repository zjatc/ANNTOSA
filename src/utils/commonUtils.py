# -*- coding: utf-8 -*-

import time
import os
import shutil
from datetime import timedelta
from hanziconv import HanziConv

''' get time difference from now to the given start timestamp '''
def get_spent_times(start_time):
    end_time = time.time()
    spent_time = end_time - start_time
    return timedelta(seconds=int(round(spent_time)))


''' common file reader, mode: 'r' for read and 'w' for write '''
def open_file(fileName, mode='r'):
    return open(fileName, mode, encoding='utf-8', errors='ignore')


def check_file_exist(filePath, fileType='', need_return=False):
    if not os.path.exists(filePath):
        if need_return:
            return False
        else:
            raise FileNotFoundError("""Given {0} file doesn't exist! {1}""" .format(fileType, filePath))
    return True

def recreate_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print('Recreated folder with path %s'%path)


def create_folder(filePath):
    folder_name = os.path.dirname(filePath)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print('Created folder with path %s'%folder_name)
    else:
        print('Folder already exists with path %s'%folder_name)


def convert_traditional_to_simple(text):
    return HanziConv.toSimplified(text)


def convert_simple_to_traditional(text):
    return HanziConv.toTraditional(x)

