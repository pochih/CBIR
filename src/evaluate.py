# -*- coding: utf-8 -*-

from __future__ import print_function

from scipy import spatial
import numpy as np


def distance(v1, v2, d_type='d1'):
  assert v1.shape == v2.shape, "shape of two vectors need to be same!"

  if d_type == 'd1':
    return np.sum(np.absolute(v1 - v2))
  elif d_type == 'd2':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd3':
    pass
  elif d_type == 'd4':
    pass
  elif d_type == 'd5':
    pass
  elif d_type == 'd6':
    pass
  elif d_type == 'd7':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd8':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'cosine':
    return spatial.distance.cosine(v1, v2)


def AP(label, results, sort=True):
  ''' infer a query, return it's ap

    arguments
      label  : query's class
      results: a dict with two keys, see the example below
               {
                 'dis': <distance between sample & query>,
                 'cls': <sample's class>
               }
      sort   : sort the results by distance
  '''
  if sort:
    results = sorted(results, key=lambda x: x['dis'])
  precision = []
  hit = 0
  for idx, result in enumerate(results):
    if result['cls'] == label:
      hit += 1
      precision.append(hit / (idx+1.))
  if hit == 0:
    return 0.
  return np.mean(precision)


def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, d_type='d1'):
  ''' infer a query, return it's ap

    arguments
      query       : a dict with two keys, see the example below
                    {
                      'img': 'orange_1.jpg',
                      'cls': 'orange'
                    }
      samples     : a list of {
                                'img': <path_to_img>,
                                'cls': <img class>,
                                'hist' <img histogram>
                              }
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  assert samples != None or (db != None and sample_db_fn != None), "need to give either samples or db plus sample_db_fn"
  if db:
    samples = sample_db_fn(db)

  q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
  results = []
  for idx, sample in enumerate(samples):
    if depth and idx >= depth:
      break
    s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
    if q_img == s_img:
      continue
    results.append({
                    'dis': distance(q_hist, s_hist, d_type=d_type),
                    'cls': s_cls
                  })
  results = sorted(results, key=lambda x: x['dis'])
  print(q_cls, results)
  ap = AP(q_cls, results, sort=False)

  return ap, results


def evaluate(db, sample_db_fn, depth=None, d_type='d1'):
  ''' infer the whole database

    arguments
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  classes = db.get_class()
  ret = {c: [] for c in classes}

  samples = sample_db_fn(db)
  for query in samples:
    ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
    ret[query['cls']].append(ap)

  return ret
