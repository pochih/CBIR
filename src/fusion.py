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

import numpy as np
import itertools
import os


d_type   = 'd1'
depth    = 30

feat_pools = ['color', 'daisy', 'edge', 'gabor', 'hog', 'vgg', 'res']

# result dir
result_dir = 'result'
if not os.path.exists(result_dir):
  os.makedirs(result_dir)


class FeatureFusion(object):

  def __init__(self, features):
    assert len(features) > 1, "need to fuse more than one feature!"
    self.features = features
    self.samples  = None

  def make_samples(self, db, verbose=False):
    if verbose:
      print("Use features {}".format(" & ".join(self.features)))

    if self.samples == None:
      feats = []
      for f_class in self.features:
        feats.append(self._get_feat(db, f_class))
      samples = self._concat_feat(db, feats)
      self.samples = samples  # cache the result
    return self.samples

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


def evaluate_feats(db, N, feat_pools=feat_pools, d_type='d1', depths=[None, 300, 200, 100, 50, 30, 10, 5, 3, 1]):
  result = open(os.path.join(result_dir, 'feature_fusion-{}-{}feats.csv'.format(d_type, N)), 'w')
  for i in range(N):
    result.write("feat{},".format(i))
  result.write("depth,distance,MMAP")
  combinations = itertools.combinations(feat_pools, N)
  for combination in combinations:
    fusion = FeatureFusion(features=list(combination))
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

  # evaluate features double-wise
  evaluate_feats(db, N=2, d_type='d1')

  # evaluate features triple-wise
  evaluate_feats(db, N=3, d_type='d1')
  
  # evaluate features quadra-wise
  evaluate_feats(db, N=4, d_type='d1')

  # evaluate features penta-wise
  evaluate_feats(db, N=5, d_type='d1')

  # evaluate features hexa-wise
  evaluate_feats(db, N=6, d_type='d1')

  # evaluate features hepta-wise
  evaluate_feats(db, N=7, d_type='d1')
  
  # evaluate database
  fusion = FeatureFusion(features=['color', 'daisy'])
  APs = evaluate_class(db, f_instance=fusion, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
