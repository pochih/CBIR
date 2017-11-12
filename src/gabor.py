# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import *
from DB import Database

from skimage.filters import gabor_kernel
from skimage import color
from scipy import ndimage as ndi

import multiprocessing

from six.moves import cPickle
import numpy as np
import scipy.misc
import os


theta     = 4
frequency = (0.1, 0.5, 0.8)
sigma     = (1, 3, 5)
bandwidth = (0.3, 0.7, 1)

n_slice  = 2
h_type   = 'global'
d_type   = 'cosine'

depth    = 1

''' MMAP
     depth
      depthNone, global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.141136758233
      depth100,  global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.216985780572
      depth30,   global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.310063286599
      depth10,   global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.3847025
      depth5,    global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.400002777778
      depth3,    global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.398166666667
      depth1,    global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.334

     (exps below use depth=None)

     _power
      gabor-global-theta4-frequency(0.1, 0.5, 0.8)-sigma(0.05, 0.25)-bandwidthNone, distance=cosine, MMAP 0.0821975313939
      gabor-global-theta6-frequency(0.1, 0.5)-sigma(1, 3)-bandwidth(0.5, 1), distance=cosine, MMAP 0.139570979988
      gabor-global-theta6-frequency(0.1, 0.8)-sigma(1, 3)-bandwidth(0.7, 1), distance=cosine, MMAP 0.139554792177
      gabor-global-theta8-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.140947344315
      gabor-global-theta6-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.139914401079
      gabor-global-theta4-frequency(0.1, 0.5, 0.8)-sigma(1, 3, 5)-bandwidth(0.3, 0.7, 1), distance=cosine, MMAP 0.141136758233
      gabor-global-theta4-frequency(0.1, 0.5, 1)-sigma(0.25, 1)-bandwidth(0.5, 1), distance=cosine, MMAP 0.120351804156
'''

def make_gabor_kernel(theta, frequency, sigma, bandwidth):
  kernels = []
  for t in range(theta):
    t = t / float(theta) * np.pi
    for f in frequency:
      if sigma:
        for s in sigma:
          kernel = gabor_kernel(f, theta=t, sigma_x=s, sigma_y=s)
          kernels.append(kernel)
      if bandwidth:
        for b in bandwidth:
          kernel = gabor_kernel(f, theta=t, bandwidth=b)
          kernels.append(kernel)
  return kernels

gabor_kernels = make_gabor_kernel(theta, frequency, sigma, bandwidth)
if sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(sigma), "kernel nums error in make_gabor_kernel()"
elif not sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * len(bandwidth), "kernel nums error in make_gabor_kernel()"
elif sigma and bandwidth:
  assert len(gabor_kernels) == theta * len(frequency) * (len(sigma) + len(bandwidth)), "kernel nums error in make_gabor_kernel()"
elif not sigma and not bandwidth:
  assert len(gabor_kernels) == theta * len(frequency), "kernel nums error in make_gabor_kernel()"

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Gabor(object):  
  
  def gabor_histogram(self, input, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size len(gabor_kernels)
        type == 'region'
          a numpy array with size len(gabor_kernels) * n_slice * n_slice
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = scipy.misc.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._gabor(img, kernels=gabor_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, len(gabor_kernels)))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._gabor(img_r, kernels=gabor_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _feats(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    feats = np.zeros(2, dtype=np.double)
    filtered = ndi.convolve(image, np.real(kernel), mode='wrap')
    feats[0] = filtered.mean()
    feats[1] = filtered.var()
    return feats
  
  
  def _power(self, image, kernel):
    '''
      arguments
        image : ndarray of the image
        kernel: a gabor kernel
      return
        a ndarray whose shape is (2, )
    '''
    image = (image - image.mean()) / image.std()  # Normalize images for better comparison.
    f_img = np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    feats = np.zeros(2, dtype=np.double)
    feats[0] = f_img.mean()
    feats[1] = f_img.var()
    return feats
  
  
  def _gabor(self, image, kernels=make_gabor_kernel(theta, frequency, sigma, bandwidth), normalize=True):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
  
    img = color.rgb2gray(image)
  
    results = []
    feat_fn = self._power
    for kernel in kernels:
      results.append(pool.apply_async(self._worker, (img, kernel, feat_fn)))
    pool.close()
    pool.join()
    
    hist = np.array([res.get() for res in results])
  
    if normalize:
      hist = hist / np.sum(hist, axis=0)
  
    return hist.T.flatten()
  
  
  def _worker(self, img, kernel, feat_fn):
    try:
      ret = feat_fn(img, kernel)
    except:
      print("return zero")
      ret = np.zeros(2)
    return ret
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "gabor-{}-theta{}-frequency{}-sigma{}-bandwidth{}".format(h_type, theta, frequency, sigma, bandwidth)
    elif h_type == 'region':
      sample_cache = "gabor-{}-n_slice{}-theta{}-frequency{}-sigma{}-bandwidth{}".format(h_type, n_slice, theta, frequency, sigma, bandwidth)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.gabor_histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


if __name__ == "__main__":
  db = Database()

  # evaluate database
  APs = evaluate_class(db, f_class=Gabor, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
