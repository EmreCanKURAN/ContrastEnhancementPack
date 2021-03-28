import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import imutil
"""
Applies logarithmic transformation on an image. Takes image, levels and c as parameters.
Returns the new image
"""
def log_transform(im, levels, c):
#create output image
    im_out = np.zeros((im.shape[0],im.shape[1],3), dtype=np.uint8)
#if image is gray
    if imutil.is_gray(im):
        im_out = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
#since opencv handles gray images like 3-channel image, reduce the channel size
        im = imutil.correct_gray(im)
#get height and width
#normalize the image to perform logarithmic transform
        im_out = im/(levels-1)
#apply the formula
        im_out = c*np.log(1+im_out)
#normalize in range 0...255
        im_out = (im_out / im_out.max()) * (levels-1)
    else:
#convert to YCbCr
        im_out = imutil.rgb2ycbcr(im)
        dMat = im_out[:, :, 0].astype(np.float)
        dMat = dMat/(levels-1)
        dMat = c*np.log(1 + (dMat))
        im_out[:, :, 0] = (dMat / dMat.max()) * (levels-1)
#convert back to RGB
        im_out = imutil.ycbcr2rgb(im_out)
    return np.uint8(im_out)

"""
Applies exponential transformation on an image. Takes image, levels and c as parameters.
Returns the new image
"""
def exp_transform(im, levels, c, a):
#create output image
    im_out = np.zeros((im.shape[0],im.shape[1],3), dtype=np.uint8)

#if image is gray
    if imutil.is_gray(im):
        im_out = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
        im = imutil.correct_gray(im)
        im_out = im/(levels-1)
        im_out = c*((1 + a)**im_out - 1)
        im_out = (im_out / im_out.max()) * (levels-1)
#if image is color
    else:
        im_out = imutil.rgb2ycbcr(im)
        dMat = im_out[:, :, 0]
        dMat = dMat/(levels-1)
        dMat = c*((1 + a)**dMat - 1)
        im_out[:, :, 0] = dMat / dMat.max() * (levels-1)
        im_out = imutil.ycbcr2rgb(im_out)
    return np.uint8(im_out)

"""
Applies power law(gamma) transformation on an image. Takes image, levels and c as parameters.
Returns the new image
"""
def pow_law_transform(im, levels, c, a):
#create output image
    im_out = np.zeros((im.shape[0],im.shape[1],3), dtype=np.uint8)
#if image is gray
    if imutil.is_gray(im):
        im_out = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
        im = imutil.correct_gray(im)
        im_out = im/(levels-1)
        im_out = c*(im_out**a)
        im_out = (im_out / im_out.max()) * (levels-1)
#if image is color
    else:
        im_out = imutil.rgb2ycbcr(im)
        dMat = im_out[:, :, 0]
        dMat = dMat/(levels-1)
        dMat = c*(dMat**a)
        im_out[:, :, 0] = dMat / dMat.max() * (levels-1)
        im_out = imutil.ycbcr2rgb(im_out)
    return np.uint8(im_out)

"""
Applies contrast stretching on an image. Takes image an levels as parameters.
Returns the new image
"""
def cont_stretch(im, levels):
#create output image
    im_out = np.zeros((im.shape[0],im.shape[1],3), dtype=np.uint8)
#compute upper and lower bounds according to formula
    a, b = 0, levels-1
    c, d = im.min(), im.max()
#get image shape height, width and num. of channels and apply formula
    if imutil.is_gray(im):
        im_out = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
        im = imutil.correct_gray(im)
        h, w = im.shape
        im_out[0:h, 0:w] = (im[0:h, 0:w] - c)*((b - a)/(d-c)) + a
#apply formula for rgb , to prevent color distortion, use the same c and d values
    else:
        h, w, ch = im.shape
        img = imutil.rgb2ycbcr(im)
        dMat = img[:,:,0]
        dMat[0:h, 0:w] = (dMat[0:h, 0:w] - c)*((b - a)/(d-c)) + a
        img[:,:,0] = dMat
        im_out = imutil.ycbcr2rgb(img)
    return im_out

"""
Applies histogram sliding on an image. Takes image, levels, value and direction(ctrl) as parameters.
Returns the new image
"""
def hist_slide(im, levels, val, ctrl):
#create the output image
    im_out = np.zeros((im.shape[0],im.shape[1],3), dtype=np.uint8)
#slide the histogram while maintaining the boundaries
    if imutil.is_gray(im):
        im = imutil.correct_gray(im)
        im_out = im.copy()
        if(ctrl == 'inc'):
            lim = 255 - val
            im_out[im_out > lim] = 255
            im_out[im_out <= lim] += val
        else:
            lim = 0 + val
            im_out[im_out < lim] = 0
            im_out[im_out >= lim] -= val
if rgb
    else:
        img = imutil.rgb2ycbcr(im)
        im_out = img.copy()
        dMat = img[:,:,0]
        if(ctrl == 'inc'):
            lim = 255 - val
            dMat[dMat > lim] = 255
            dMat[dMat <= lim] += val
        else:
            lim = 0 + val
            dMat[dMat < lim] = 0
            dMat[dMat >= lim] -= val
        im_out[:,:,0] = dMat
        im_out = imutil.ycbcr2rgb(im_out)
    return np.uint8(im_out)
'''
Equalizes histogram of an image with k levels. Takes image and number of levels as parameters.
Returns the new image.
'''
def hist_eq(im, levels):
    h, w = im.shape
    tot_pixs = h * w
    im_hist = np.zeros((levels))
#count number of pixels for each level
    for i in range(0,levels):
        im_hist[i] = np.count_nonzero(im == i)
#find pdf
    pdf = np.zeros((levels))
    for i in range(0,levels):
        pdf[i] = im_hist[i]/tot_pixs
#find cdf
    cdf = np.zeros((levels))
    cdf[0] = pdf[0]
    for i in range(1, levels):
        cdf[i] = pdf[i] + cdf[i-1]
    im2 = np.zeros((h,w))
#transform
    for j in range(0,h):
        for k in range(0,w):
            im2[j, k] = int(round((levels-1) * cdf[im[j, k]]))
    return np.uint8(im2)
"""
Utility for hist_eq, handles the cases for gray and RGB images
Returns the new image
"""
def hist_eq_util(im, levels):
#detect if image is gray or rgb, do appropriate operations
    im_out = im.copy()
    if imutil.is_gray(im):
        im = imutil.correct_gray(im)
        im_out = hist_eq(im, levels)
    else:
        im_out = imutil.rgb2ycbcr(im)
        im_out[:,:,0] = hist_eq(im_out[:,:,0], levels)
        im_out = imutil.ycbcr2rgb(im_out)
    return im_out

'''
Equalizes histogram of an image with k levels using BBHE algorithm. Takes image and number of levels as parameters.
Returns the new image.
'''
def bbheq(im, levels):
    h, w = im.shape
    tot_pixs = h * w
#same as computing via for loops
    mean_im = im.mean()
#round it up and convert to int
    mean_im = int(round(mean_im))
#create hashmaps with ranges where each pixel can be 0 or 1
    im_l_hash = (im <= mean_im)
    im_u_hash = (im > mean_im)
#count the number of pixels for lower part and upper part
    im_l_pix_count = np.count_nonzero(im_l_hash)
    im_u_pix_count = np.count_nonzero(im_u_hash)
#allocate histograms
    im_l_hist = np.zeros(((mean_im) + 1))
    im_u_hist = np.zeros(((levels-1)-mean_im))
#create histograms according to formula
    for i in range(0,mean_im + 1):
        im_l_hist[i] = np.count_nonzero(im == i)
    for i in range(mean_im + 1, levels):
        im_u_hist[i-mean_im-1] = np.count_nonzero(im == i)
#allocate pdfs
    pdf_l = np.zeros(((mean_im) + 1))
    pdf_u = np.zeros(((levels-1)-mean_im))
#find pdfs
    for i in range(0,mean_im + 1):
        pdf_l[i] = im_l_hist[i]/im_l_pix_count
    for i in range(mean_im + 1, levels):
        pdf_u[i-mean_im-1] = im_u_hist[i-mean_im-1]/im_u_pix_count
#allocate cdfs
    cdf_l = np.zeros(((mean_im) + 1))
    cdf_u = np.zeros(((levels-1)-mean_im))
#find cdfs
    cdf_l[0] = pdf_l[0]
    for i in range(1,mean_im + 1):
        cdf_l[i] = pdf_l[i] + cdf_l[i-1]
    cdf_u[0] = pdf_u[0]
    for i in range(mean_im + 1, levels):
        cdf_u[i-mean_im-1] = pdf_u[i-mean_im-1] + cdf_u[i-mean_im-2]
#transform
    im2 = np.zeros((h,w))
    for j in range(0,h):
        for k in range(0,w):
            if(im_l_hash[j, k] == 1):
                im2[j, k] = int(round((mean_im))) * cdf_l[im[j, k]]
            elif(im_u_hash[j, k] == 1):
                im2[j, k] = int(round((mean_im) + 1)) + ((levels-1) - int(round(mean_im)+1)) * cdf_u[im[j, k]-mean_im-1]
    return im2
"""
Utility for bbheq, handles the cases for gray and RGB images
Returns the new image
"""
def bbheq_util(im, levels):
    im_out = im.copy()
    if imutil.is_gray(im):
        im = imutil.correct_gray(im)
        im_out = bbheq(im, levels)
    else:
        im_out = imutil.rgb2ycbcr(im)
        im_out[:,:,0] = bbheq(im_out[:,:,0], levels)
        im_out = imutil.ycbcr2rgb(im_out)
    return im_out

'''
Equalizes histogram of an image with k levels using DSIHE algorithm. Takes image and number of levels as parameters.
Returns the new image.
'''
def dsiheq(im, levels):
    h, w = im.shape
    tot_pixs = h * w
#allocate histogram, calculate pdf and cdf as done in classical HE algorithm
    im_hist = np.zeros((levels))
    for i in range(0,levels):
        im_hist[i] = np.count_nonzero(im == i)
    pdf = np.zeros((levels))
    for i in range(0,levels):
        pdf[i] = im_hist[i]/tot_pixs
    cdf = np.zeros((levels))
    cdf[0] = pdf[0]
    for i in range(1, levels):
        cdf[i] = pdf[i] + cdf[i-1]
    median = 0
    argmin_d = 1
#find the median according to the formula
    for i in range(0, levels):
        if abs(cdf[i]-((cdf[levels-1]+cdf[0])/2))<argmin_d:
            argmin_d = abs(cdf[i]-((cdf[levels-1]+cdf[0])/2))
            median = i
#same steps in BBHE, now divide image to two parts using median value and keep hashmaps
    im_l_hash = (im <= median - 1)
    im_u_hash = (im > median - 1)
#count number of pixels for each part
    im_l_pix_count = np.count_nonzero(im_l_hash)
    im_u_pix_count = np.count_nonzero(im_u_hash)
#allocate histograms
    im_l_hist = np.zeros((median))
    im_u_hist = np.zeros(((levels)-median))
#find histograms
    for i in range(0,median):
        im_l_hist[i] = np.count_nonzero(im == i)
    for i in range(median, levels):
        im_u_hist[i-median] = np.count_nonzero(im == i)
#allocate pdfs
    pdf_l = np.zeros((median))
    pdf_u = np.zeros(((levels)-median))

#find pdfs
    for i in range(0,median):
        pdf_l[i] = im_l_hist[i]/im_l_pix_count
    for i in range(median, levels):
        pdf_u[i-median] = im_u_hist[i-median]/im_u_pix_count
#allocate cdfs
    cdf_l = np.zeros((median))
    cdf_u = np.zeros(((levels)-median))
#find cdffs
    cdf_l[0] = pdf_l[0]
    for i in range(1,median):
        cdf_l[i] = pdf_l[i] + cdf_l[i-1]
    cdf_u[0] = pdf_u[0]
    for i in range(median + 1, levels):
        cdf_u[i-median] = pdf_u[i-median] + cdf_u[i-median-1]
#transform
    im2 = np.zeros((h,w))
    for j in range(0,h):
        for k in range(0,w):
            if(im_l_hash[j, k] == 1):
                im2[j, k] = int(round((median-1))) * cdf_l[im[j, k]]
            elif(im_u_hash[j, k] == 1):
                im2[j, k] = int(round((median))) +  ((levels-1) - int(round(median))) * cdf_u[im[j, k]-median]
    return im2
"""
Utility for dsiheq, handles the cases for gray and RGB images
Returns the new image
"""
def dsiheq_util(im, levels):
#get number of levels and copy the original image
    im_out = im.copy()
#get image shape height, width and num. of channels
    if imutil.is_gray(im):
        im = imutil.correct_gray(im)
        im_out = dsiheq(im, levels)
#apply formula, to prevent color distortion, use the same c and d values
    else:
        im_out = imutil.rgb2ycbcr(im)
        im_out[:,:,0] = dsiheq(im_out[:,:,0], levels)
        im_out = imutil.ycbcr2rgb(im_out)
    return im_out
