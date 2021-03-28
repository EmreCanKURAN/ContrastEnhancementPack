import cv2
from matplotlib import pyplot as plt
import math
import imutil
import cont_util
import numpy as np

"""
Main function
"""
def main():
#read images , get levels and results image
    im = cv2.imread('test_im/4.2.03.tiff')
    levels = imutil.get_depth(im)
#    out = cont_util.log_transform(im, levels, 2)
#    out = cont_util.exp_transform(im, levels, 4, 0.5)
#    out = cont_util.pow_law_transform(im, levels, 2, 0.5)
#    out = cont_util.hist_slide(im, levels, 150, 'dec')
#    out = cont_util.hist_slide(im2, levels, 50, 'inc')
#    out = cont_util.cont_stretch(im, levels)
#    out = cont_util.hist_eq_util(im, levels)
    out = cont_util.bbheq_util(im, levels)
#    out = cont_util.dsiheq_util(im, levels)
#give the names for the titles of the images
    names = ['Original', 'BBHE']
#create image list
    im_list = [im,out]
#show images and their histograms
    imutil.im_plot(im_list , names , levels)
    imutil.hist_plot(im_list , names , levels)
if __name__ == "__main__":
    main()
