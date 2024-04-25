import os
import numpy as np
import pandas as pd
import numpy as np
import math
import copy
import time
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, SqrtStretch
import matplotlib.pyplot as plt
import sys
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec

def getImgStats(img):
    v = list(np.percentile(img, [10, 25, 50, 75, 90, 99]))
    s = f"\navg:\t{np.mean(img)}"
    s += f"\nstd:\t{np.std(img)}"
    s += f"\nmin:\t{np.min(img)}"
    s += f"\n10:\t{v[0]}"
    s += f"\n25:\t{v[1]}"
    s += f"\n50:\t{v[2]}"
    s += f"\n75:\t{v[3]}"
    s += f"\n90:\t{v[4]}"
    s += f"\n99:\t{v[5]}"
    s += f"\nmax:\t{np.max(img)}\n"
    return s


def inspect_data(slim=True, colorbar=True):
    if slim:
        data_path = './slim_LAII.fits'
        model_path = './model_slim_LAII.fits'
    else:
        data_path = './LAII.fits'
        model_path = './model_LAII.fits'
    ax1,ax2 = inspect_moments(data_path, colorbar)
    ax3,ax4 = inspect_moments(model_path, colorbar)
    ax5,ax6 = inspect_diffs(data_path, model_path, colorbar)
    ax7,ax8 = inspect_hists(data_path)
    ax9,ax10 = inspect_hists(model_path)
    plt.show()
    plt.close()
    return


def inspect_diffs(data_path, model_path, colorbar):
    data = fits.getdata(data_path)
    model_data = fits.getdata(model_path)
    diff_1 = np.sum(data, axis = (1)) - np.sum(model_data, axis = (1))
    diff_2 = np.sum(data, axis = (2)) - np.sum(model_data, axis = (2))
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(diff_1)
    ax1.set_title('First Moment: Data - Model')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Velocity')
    if colorbar:
        plt.colorbar(im1, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(diff_2)
    ax2.set_title('Second Moment: Data - Model')
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Velocity')
    if colorbar:
        plt.colorbar(im2, ax=ax2)
    return ax1, ax2

def inspect_hists(path):
    
    data = fits.getdata(path)
    first = np.sum(data, axis = (1))
    second = np.sum(data, axis = (2))
    
    hist_range=None
    bins = 500
    log = True

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(first.flatten(), bins=bins, range=hist_range, log=log, color='blue', alpha=0.7)
    ax1.set_title('Histogram of First Moment')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(second.flatten(), bins=bins, range=hist_range, log=log, color='blue', alpha=0.7)
    ax2.set_title('Histogram of Second Moment')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    return ax1, ax2


def inspect_moments(path, colorbar):
    data = fits.getdata(path)
    first = np.sum(data, axis = (1))
    second = np.sum(data, axis = (2))
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.imshow(first, vmin=0, vmax=10)
    ax1.set_title('First Moment')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Velocity')
    if colorbar:
        plt.colorbar(im1, ax=ax1)
    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(second, vmin=0, vmax=10)
    ax2.set_title('Second Moment')
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Velocity')
    if colorbar:
        plt.colorbar(im2, ax=ax2)
    return ax1, ax2


def newZScale(img):
    print('computing new zscale')
    zscale = ZScaleInterval(contrast=0.3)
    vmin, vmax = zscale.get_limits(img)
    norm = ImageNormalize(vmin=vmin, vmax=vmax)
    return norm(img)

def logstretch(img):
    stretch = LogStretch()
    return stretch(img)

def sqrtstretch(img):
    stretch = SqrtStretch()
    return stretch(img)

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))
