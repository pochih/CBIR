# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

from six.moves import cPickle
import numpy as np
import scipy.misc
from math import sqrt
import os


stride = (1, 1)
n_slice  = 10
h_type   = 'region'
d_type   = 'cosine'

depth    = 5

''' MMAP    
      depth
       depthNone, region-stride(1, 1)-n_slice10,co, MMAP 0.101670982288
       depth100,  region-stride(1, 1)-n_slice10,co, MMAP 0.207817305128
       depth30,   region-stride(1, 1)-n_slice10,co, MMAP 0.291715090839
       depth10,   region-stride(1, 1)-n_slice10,co, MMAP 0.353722379063
       depth5,    region-stride(1, 1)-n_slice10,co, MMAP 0.367119444444
       depth3,    region-stride(1, 1)-n_slice10,co, MMAP 0.3585
       depth1,    region-stride(1, 1)-n_slice10,co, MMAP 0.302
  
       (exps below use depth=None)
  
      d_type
       global-stride(2, 2),d1, MMAP 0.0530993236031
       global-stride(2, 2),co, MMAP 0.0528310744618
  
      stride
       region-stride(2, 2)-n_slice4,d1, MMAP 0.0736245142237
       region-stride(1, 1)-n_slice4,d1, MMAP 0.0704206226545
  
      n_slice
       region-stride(1, 1)-n_slice10,co, MMAP 0.101670982288
       region-stride(1, 1)-n_slice6,co, MMAP 0.0977736743859
  
      h_type
       global-stride(2, 2),d1, MMAP 0.0530993236031
       region-stride(2, 2)-n_slice4,d1, MMAP 0.0736245142237
'''

edge_kernels = np.array([
  [
   # vertical
   [1,-1], 
   [1,-1]
  ],
  [
   # horizontal
   [1,1], 
   [-1,-1]
  ],
  [
   # 45 diagonal
   [sqrt(2),0], 
   [0,-sqrt(2)]
  ],
  [
   # 135 diagnol
   [0,sqrt(2)], 
   [-sqrt(2),0]
  ],
  [
   # non-directional
   [2,-2], 
   [-2,2]
  ]
])

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class Edge(object):

  def histogram(self, input, stride=(2, 2), type=h_type, n_slice=n_slice, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        stride   : stride of edge kernel
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size len(edge_kernels)
        type == 'region'
          a numpy array with size len(edge_kernels) * n_slice * n_slice
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = scipy.misc.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._conv(img, stride=stride, kernels=edge_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, edge_kernels.shape[0]))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._conv(img_r, stride=stride, kernels=edge_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _conv(self, img, stride, kernels, normalize=True):
    H, W, C = img.shape
    conv_kernels = np.expand_dims(kernels, axis=3)
    conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
    assert list(conv_kernels.shape) == list(kernels.shape) + [C]  # check kernels size
  
    sh, sw = stride
    kn, kh, kw, kc = conv_kernels.shape
  
    hh = int((H - kh) / sh + 1)
    ww = int((W - kw) / sw + 1)
  
    hist = np.zeros(kn)
  
    for idx, k in enumerate(conv_kernels):
      for h in range(hh):
        hs = int(h*sh)
        he = int(h*sh + kh)
        for w in range(ww):
          ws = w*sw
          we = w*sw + kw
          hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "edge-{}-stride{}".format(h_type, stride)
    elif h_type == 'region':
      sample_cache = "edge-{}-stride{}-n_slice{}".format(h_type, stride, n_slice)
  
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
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


if __name__ == "__main__":
  db = Database()

  # check shape
  assert edge_kernels.shape == (5, 2, 2)

  # evaluate database
  APs = evaluate_class(db, f_class=Edge, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
