# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import *
from DB import Database
from six.moves import cPickle
import numpy as np
import scipy.misc
import itertools
import os

# configs for histogram
n_bin    = 4
n_slice  = 4
h_type   = 'region'
d_type   = 'd1'

''' MMAP
      bin4,slice10,MMAP 0.232772046495
      bin6,slcie3,MMAP 0.262819311357
      bin8,slice6,MMAP 0.25908443026
      bin10,slice4,MMAP 0.269872790396
      bin12,slice2,MMAP 0.266076627332
      bin12,slice3,MMAP 0.273745840034
      bin12,slice4,MMAP 0.271520862017
      bin14,slice3,MMAP 0.272386552594
      bin14,slice5,MMAP 0.266877181379
      bin16,slice3,MMAP 0.273716788003
      bin16,slice4,MMAP 0.272221031804
      bin16,slice8,MMAP 0.253823360098
'''

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


def histogram(input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
  ''' count img histogram

    arguments
      input    : a path to a image or a numpy.ndarray
      n_bin    : number of bins for each channel
      type     : 'global' means count the histogram for whole image
                 'region' means count the histogram for regions in images, then concatanate all of them
      n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
      normalize: normalize output histogram
  '''
  try:
    assert isinstance(input, np.ndarray)  # examinate input type
    img = input.copy()
  except:
    img = scipy.misc.imread(input, mode='RGB')
  height, width, channel = img.shape
  bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel

  if type == 'global':
    hist = _count_hist(img, n_bin, bins, channel)

  elif type == 'region':
    hist = np.zeros((n_slice, n_slice, n_bin ** channel))
    h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
    w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)

    for hs in range(len(h_silce)-1):
      for ws in range(len(w_slice)-1):
        img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
        hist[hs][ws] = _count_hist(img_r, n_bin, bins, channel)

  if normalize:
    hist /= np.sum(hist)

  return hist.flatten()


def _count_hist(input, n_bin, bins, channel):
  img = input.copy()
  bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
  hist = np.zeros(n_bin ** channel)

  # cluster every pixels
  for idx in range(len(bins)-1):
    img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
  # add pixels into bins
  height, width, _ = img.shape
  for h in range(height):
    for w in range(width):
      b_idx = bins_idx[tuple(img[h,w])]
      hist[b_idx] += 1

  return hist


def make_sample(db, verbose=True):
  if h_type == 'global':
    sample_cache = "histogram_cache-{}-n_bin{}".format(h_type, n_bin)
  elif h_type == 'region':
    sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(h_type, n_bin, n_slice)
  
  try:
    samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
    if verbose:
      print("Using cache..., config=%s" % sample_cache)
  except:
    if verbose:
      print("Counting histogram..., config=%s" % sample_cache)
    samples = []
    data = db.get_data()
    for d in data.itertuples():
      d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
      d_hist = histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
      samples.append({
                      'img':  d_img, 
                      'cls':  d_cls, 
                      'hist': d_hist
                    })
    cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))

  return samples


if __name__ == "__main__":
  db = Database()
  data = db.get_data()

  # test normalize
  hist = histogram(data.ix[0,0], type='global')
  assert hist.sum() - 1 < 1e-9

  # test histogram bins
  def sigmoid(z):
    a = 1.0 / (1.0 + np.exp(-1. * z))
    return a
  np.random.seed(0)
  IMG = sigmoid(np.random.randn(2,2,3)) * 255
  IMG = IMG.astype(int)
  hist = histogram(IMG, type='global', n_bin=4)
  assert np.equal(np.where(hist > 0)[0], np.array([37, 43, 58, 61])).all()  # judge global histogram
  hist = histogram(IMG, type='region', n_bin=4, n_slice=2)
  assert np.equal(np.where(hist > 0)[0], np.array([58, 125, 165, 235])).all()  # judge region histogram

  # examinate distance
  np.random.seed(1)
  IMG = sigmoid(np.random.randn(4,4,3)) * 255
  IMG = IMG.astype(int)
  hist = histogram(IMG, type='region', n_bin=4, n_slice=2)
  IMG2 = sigmoid(np.random.randn(4,4,3)) * 255
  IMG2 = IMG2.astype(int)
  hist2 = histogram(IMG2, type='region', n_bin=4, n_slice=2)
  assert distance(hist, hist2, type='d1') == 2
  assert distance(hist, hist2, type='d2') == 2

  # evaluate database
  APs = evaluate(db, sample_db_fn=make_sample)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
