import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
'''
Takes image as input, reduces channel size to one if image is gray.Returns a gray image.
'''
def correct_gray(im):
#if channel size is normal(only width and height) return image
    if len(im.shape) < 3 or im.shape[2]  == 1:
        return im
#if channel size >= 3 fix it by converting to gray using classical rgb to gray formula
    else:
        return im[:,:,0]
'''
Takes image as input, converts rgb to gray. Return gray image.
'''
def bgr2gray(im):
    return im[:,:,2]*0.1140 + im[:,:,1]*0.5870 + im[:,:,0]*0.2989
'''
Takes image as input, controls if image is gray or not. Returns true or false.
'''
def is_gray(im):
#if image is gray, return true
    if len(im.shape) < 3 or im.shape[2]  == 1:
        return True
#get the channels and check them, if all the channels are equal then it is a grayscale image
    b,g,r = im[:,:,0], im[:,:,1], im[:,:,2]
    if (b==g).all() and (b==r).all():
        return True
#else it is not gray
    return False

'''
Converts BGR image to RGB and returns the new RGB image
'''
def bgr2rgb(im):
    h, w, ch = im.shape
    new_im = np.zeros((h, w, ch))
    new_im[:,:,0], new_im[:,:,1], new_im[:,:,2] = im[:,:,2], im[:,:,1], im[:,:,0]
    return np.uint8(new_im)
'''
Takes image as input, converts image to YCbCr using RGB->YCrCb formula. Returns new YUV image.
'''
def rgb2ycbcr(im):
#define coefficients as a numpy array
    coef = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
#get the dot product of image and coefficients for conversion
    ycbcr = im.dot(coef.T)
#Add 128 to Cr and Cb channels
    ycbcr[:,:,[1,2]] += 128
#return new YCrCb image
    return np.uint8(ycbcr)
'''
Takes image as input, converts image to RGB using YUV->RGB formula. Returns new BGR image.
'''
def ycbcr2rgb(im):
#define coefficients as a numpy array
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
#convert to float in order to prevent computation errors
    rgb = im.astype(np.float)
#Subtract 128 from Cr and Cb channels
    rgb[:,:,[1,2]] -= 128
#get the dot product of image and coefficients for conversion
    rgb = rgb.dot(xform.T)
#if the values are bigger than 255 change it as 255
    np.putmask(rgb, rgb > 255, 255)
#if the values are smaller than 0 change it as 0
    np.putmask(rgb, rgb < 0, 0)
#return RGB image
    return np.uint8(rgb)

'''
Takes the image list, image names and num. of levels as input
Plots the images, returns nothing.
'''
def im_plot(im_list, names, levels):
#define figures, axes and image counter i
    fig, ax = plt.subplots(1,len(im_list),figsize=(10, 8))
    i = 0
#show images together using subplot
    for im in im_list:
        i = i + 1
#if gray, handle
        if is_gray(im):
            im_new = im
            plt_str = int("1" + str(len(im_list)) + str(i))
            plt.subplot(plt_str)
#map it to gray
            plt.imshow(im_new, cmap='gray')
            plt.title(names[i-1])
            plt.axis('off')
#if color, handle
        else:
            im_new = bgr2rgb(im)
            plt_str = int("1" + str(len(im_list)) + str(i))
            plt.subplot(plt_str)
            plt.imshow(im_new)
            plt.title(names[i-1])
            plt.axis('off')
#show
    plt.show()

'''
Takes the image list, image names and num. of levels as input
Plots the image histograms, returns nothing.
'''
def hist_plot(im_list, names, levels):
#define figures, axes and image counter i
    fig, ax = plt.subplots(1,len(im_list),figsize=(10, 8))
    i = 0
#show histograms of the images together using subplot
    if is_gray(im_list[0]):
        for im in im_list:
            i = i + 1
            plt_str = int("1" + str(len(im_list)) + str(i))
            plt.subplot(plt_str)
#use plt.hist for histograms
            plt.hist(im_list[i-1].ravel(), levels,[0, levels])
            plt.title(names[i-1])
#limit max and min values
            plt.xlim([0,levels])
        plt.show()
#if it is a color image, take the 3 channels
    else:    
        color = ('b','g','r')
        for j in range(0, len(im_list)):
            for i,col in enumerate(color):
                plt_str = int("1" + str(len(im_list)) + str(j+1))
                plt.subplot(plt_str)
                histr = cv2.calcHist([im_list[j]],[i],None,[levels],[0,levels])
                plt.plot(histr,color = col)
                plt.xlim([0,levels])
                plt.title(names[j])
    plt.show()
    
"""
Takes the image as the input and returns the depth(8 bit, 16 bit etc.)
"""
def get_depth(im):
    deptStr = im.dtype
#if type is uintX then the depth is X(like uint8,uint16 etc.), so we split it
    return (2**int((str(deptStr).rsplit('uint',1))[1]))
