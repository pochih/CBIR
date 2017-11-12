# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

from skimage.feature import hog
from skimage import color

from six.moves import cPickle
import numpy as np
import scipy.misc
import os

n_bin    = 10
n_slice  = 6
n_orient = 8
p_p_c    = (2, 2)
c_p_b    = (1, 1)
h_type   = 'region'
d_type   = 'd1'

depth    = 5

''' MMAP
     depth
      depthNone, HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.155887235348
      depth100,  HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.261149622088
      depth30,   HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.371054105819
      depth10,   HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.449627835097
      depth5,    HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.465333333333
      depth3,    HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.463833333333
      depth1,    HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.398

      (exps below use depth=None)

     ppc & cpb
      HOG-global-n_bin10-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.105569494513
      HOG-global-n_bin10-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.0945343258574
      HOG-global-n_bin10-n_orient8-ppc(8, 8)-cpb(3, 3), distance=d1, MMAP 0.0782408187317

     h_type
      HOG-global-n_bin100-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.0990826443803
      HOG-region-n_bin100-n_slice4-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.131164310773

     n_orient
      HOG-global-n_bin10-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.105569494513
      HOG-region-n_bin10-n_slice4-n_orient18-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.14941454752

     n_bin
      HOG-region-n_bin5-n_slice4-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.140448910465
      HOG-region-n_bin10-n_slice4-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.144675311048
      HOG-region-n_bin20-n_slice4-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.1429074023
      HOG-region-n_bin100-n_slice4-n_orient8-ppc(32, 32)-cpb(1, 1), distance=d1, MMAP 0.131164310773

     n_slice
      HOG-region-n_bin10-n_slice2-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.116513458785
      HOG-region-n_bin10-n_slice4-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.151557545391
      HOG-region-n_bin10-n_slice6-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.155887235348
      HOG-region-n_bin10-n_slice8-n_orient8-ppc(2, 2)-cpb(1, 1), distance=d1, MMAP 0.15347983005
'''

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class HOG(object):

  def histogram(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        n_bin    : number of bins of histogram
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size n_bin
        type == 'region'
          a numpy array with size n_bin * n_slice * n_slice
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = scipy.misc.imread(input, mode='RGB')
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._HOG(img, n_bin)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._HOG(img_r, n_bin)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()

  def _HOG(self, img, n_bin, normalize=True):
    image = color.rgb2gray(img)
    fd = hog(image, orientations=n_orient, pixels_per_cell=p_p_c, cells_per_block=c_p_b)
    bins = np.linspace(0, np.max(fd), n_bin+1, endpoint=True)
    hist, _ = np.histogram(fd, bins=bins)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist

  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "HOG-{}-n_bin{}-n_orient{}-ppc{}-cpb{}".format(h_type, n_bin, n_orient, p_p_c, c_p_b)
    elif h_type == 'region':
      sample_cache = "HOG-{}-n_bin{}-n_slice{}-n_orient{}-ppc{}-cpb{}".format(h_type, n_bin, n_slice, n_orient, p_p_c, c_p_b)
  
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

  # evaluate database
  APs = evaluate_class(db, f_class=HOG, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
