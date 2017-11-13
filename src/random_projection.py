# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat

from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn import random_projection
import numpy as np
import itertools
import os


feat_pools = ['color', 'daisy', 'edge', 'gabor', 'hog', 'vgg', 'res']

keep_rate = 0.25
project_type = 'sparse'

# result dir
result_dir = 'result'
if not os.path.exists(result_dir):
  os.makedirs(result_dir)


class RandomProjection(object):

  def __init__(self, features, keep_rate=keep_rate, project_type=project_type):
    assert len(features) > 0, "need to give at least one feature!"
    self.features     = features
    self.keep_rate    = keep_rate
    self.project_type = project_type

    self.samples      = None

  def make_samples(self, db, verbose=False):
    if verbose:
      print("Use features {}, {} RandomProject, keep {}".format(" & ".join(self.features), self.project_type, self.keep_rate))

    if self.samples == None:
      feats = []
      for f_class in self.features:
        feats.append(self._get_feat(db, f_class))
      samples = self._concat_feat(db, feats)
      samples, _ = self._rp(samples)
      self.samples = samples  # cache the result
    return self.samples

  def check_random_projection(self):
    ''' check if current smaple can fit to random project

       return
         a boolean
    '''
    if self.samples == None:
      feats = []
      for f_class in self.features:
        feats.append(self._get_feat(db, f_class))
      samples = self._concat_feat(db, feats)
      samples, flag = self._rp(samples)
      self.samples = samples  # cache the result
    return True if flag else False

  def _get_feat(self, db, f_class):
    if f_class == 'color':
      f_c = Color()
    elif f_class == 'daisy':
      f_c = Daisy()
    elif f_class == 'edge':
      f_c = Edge()
    elif f_class == 'gabor':
      f_c = Gabor()
    elif f_class == 'hog':
      f_c = HOG()
    elif f_class == 'vgg':
      f_c = VGGNetFeat()
    elif f_class == 'res':
      f_c = ResNetFeat()
    return f_c.make_samples(db, verbose=False)

  def _concat_feat(self, db, feats):
    samples = feats[0]
    delete_idx = []
    for idx in range(len(samples)):
      for feat in feats[1:]:
        feat = self._to_dict(feat)
        key = samples[idx]['img']
        if key not in feat:
          delete_idx.append(idx)
          continue
        assert feat[key]['cls'] == samples[idx]['cls']
        samples[idx]['hist'] = np.append(samples[idx]['hist'], feat[key]['hist'])
    for d_idx in sorted(set(delete_idx), reverse=True):
      del samples[d_idx]
    if delete_idx != []:
      print("Ignore %d samples" % len(set(delete_idx)))

    return samples

  def _to_dict(self, feat):
    ret = {}
    for f in feat:
      ret[f['img']] = {
        'cls': f['cls'],
        'hist': f['hist']
      }
    return ret

  def _rp(self, samples):
    feats = np.array([s['hist'] for s in samples])
    eps = self._get_eps(n_samples=feats.shape[0], n_dims=feats.shape[1])
    if eps == -1:
      import warnings
      warnings.warn(
        "Can't fit to random projection with keep_rate {}\n".format(self.keep_rate), RuntimeWarning
      )
      return samples, False
    if self.project_type == 'gaussian':
      transformer = random_projection.GaussianRandomProjection(eps=eps) 
    elif self.project_type == 'sparse':
      transformer = random_projection.SparseRandomProjection(eps=eps)
    feats = transformer.fit_transform(feats)
    assert feats.shape[0] == len(samples)
    for idx in range(len(samples)):
      samples[idx]['hist'] = feats[idx]
    return samples, True

  def _get_eps(self, n_samples, n_dims, n_slice=int(1e4)):
    new_dim = n_dims * self.keep_rate
    for i in range(1, n_slice):
      eps = i / n_slice
      jl_dim = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
      if jl_dim <= new_dim:
        print("rate %.3f, n_dims %d, new_dim %d, dims error rate: %.4f" % (self.keep_rate, n_dims, jl_dim, ((new_dim-jl_dim) / new_dim)) )
        return eps
    return -1


def evaluate_feats(db, N, feat_pools=feat_pools, keep_rate=keep_rate, project_type=project_type, d_type='d1', depths=[None, 300, 200, 100, 50, 30, 10, 5, 3, 1]):
  result = open(os.path.join(result_dir, 'feature_reduction-{}-keep{}-{}-{}feats.csv'.format(project_type, keep_rate, d_type, N)), 'w')
  for i in range(N):
    result.write("feat{},".format(i))
  result.write("depth,distance,MMAP")
  combinations = itertools.combinations(feat_pools, N)
  for combination in combinations:
    fusion = RandomProjection(features=list(combination), keep_rate=keep_rate, project_type=project_type)
    if fusion.check_random_projection():
      for d in depths:
        APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=d)
        cls_MAPs = []
        for cls, cls_APs in APs.items():
          MAP = np.mean(cls_APs)
          cls_MAPs.append(MAP)
        r = "{},{},{},{}".format(",".join(combination), d, d_type, np.mean(cls_MAPs))
        print(r)
        result.write('\n'+r)
      print()
  result.close()


if __name__ == "__main__":
  db = Database()

  # evaluate features single-wise
  evaluate_feats(db, N=1, d_type='d1', keep_rate=keep_rate, project_type=project_type)

  # evaluate features double-wise
  evaluate_feats(db, N=2, d_type='d1', keep_rate=keep_rate, project_type=project_type)

  # evaluate features triple-wise
  evaluate_feats(db, N=3, d_type='d1', keep_rate=keep_rate, project_type=project_type)
  
  # evaluate features quadra-wise
  evaluate_feats(db, N=4, d_type='d1', keep_rate=keep_rate, project_type=project_type)

  # evaluate features penta-wise
  evaluate_feats(db, N=5, d_type='d1', keep_rate=keep_rate, project_type=project_type)

  # evaluate features hexa-wise
  evaluate_feats(db, N=6, d_type='d1', keep_rate=keep_rate, project_type=project_type)

  # evaluate features hepta-wise
  evaluate_feats(db, N=7, d_type='d1', keep_rate=keep_rate, project_type=project_type)
  
  # evaluate color feature
  d_type = 'd1'
  depth  = 30
  fusion = RandomProjection(features=['color'], keep_rate=keep_rate, project_type=project_type)
  APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
