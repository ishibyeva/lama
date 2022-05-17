import os
import subprocess
import sys
import time

import cv2
import pytest
from skimage.metrics import structural_similarity

parent_dir = os.getcwd()

def test_check_model():
    is_testing_materials_here = os.path.isdir(os.path.join(parent_dir, 'big-lama'))
    assert is_testing_materials_here == True, 'Failed: test directory not found'

def test_inpainting_with_standart_set_with_masks():
    #python3 bin/predict.py model.path=$(pwd)/big-lama 
    # indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output
    #model_dir = os.path.join(parent_dir, 'big-lama')
    #testing_images = os.path.join(parent_dir, 'LaMa_test_images')
    #output_dir = os.path.join(parent_dir, 'output')
    
    start_predict = ['python3', 'bin/predict.py', 'model.path=' + sys.argv[2],
                    'indir=' + sys.argv[3], 'outdir=' + os.path.join(parent_dir, 'output')]

    start_time = time.time()
    proc = subprocess.Popen(start_predict, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
    end_time = time.time()
    out,err = proc.communicate()
    assert end_time - start_time < 1500.0, 'Failed: inpainting timeout'

def test_check_directory_exists():
    is_output_here = os.path.isdir(os.path.join(parent_dir, 'output'))
    assert is_output_here == True, 'Failed: directory with output files not found'

def test_check_output_pictures():
    output_dir = os.path.join(parent_dir, 'output')
    files_list = os.listdir(output_dir)
    assert len(files_list) != 0, 'Failed: result directory is empty'

def test_compare_pictures_with_original():
    output_dir = os.path.join(parent_dir, 'output')
    output_files = os.listdir(output_dir)
    compare_dir = os.path.join(parent_dir, 'compare_out') 
    compare_files = os.listdir(compare_dir)
    ssim = []
    
    for image in compare_files:
        imageA = cv2.imread(os.path.join(output_dir, image))
        imageB = cv2.imread(os.path.join(compare_dir, image))
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        diff = structural_similarity(grayA, grayB)
        ssim.append(diff)
        
    difference_indicator = any(index < 0.98 for index in ssim)
    assert difference_indicator == False, 'Failed: wrong image inpainting result'
