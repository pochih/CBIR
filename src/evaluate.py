# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

def distance(v1, v2, type='d1'):
  assert v1.shape == v2.shape

  if type == 'd1':
    # return 1 - np.minimum(v1, v2) / np.minimum(v1.size, v2.size)
    return np.sum(np.absolute(v1 - v2))
  elif type == 'd2':
    return 2 - 2 * np.dot(v1, v2)
  elif type == 'd3':
    pass
  elif type == 'd4':
    pass
  elif type == 'd5':
    pass
  elif type == 'd6':
    pass
  elif type == 'd7':
    return 2 - 2 * np.dot(v1, v2)
  elif type == 'd8':
    return 2 - 2 * np.dot(v1, v2)


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
  return np.mean(precision)


def infer(query, samples=None, db=None, sample_db_fn=None):
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
  '''
  assert samples != None or (db != None and sample_db_fn != None)
  if db:
    samples = sample_db_fn(db)

  q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
  results = []
  for sample in samples:
    s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
    if q_img == s_img:
      continue
    results.append({
                    'dis': distance(q_hist, s_hist),
                    'cls': s_cls
                  })
  results = sorted(results, key=lambda x: x['dis'])
  ap = AP(q_cls, results, sort=False)

  return ap, results


def evaluate(db, sample_db_fn):
  ''' infer the whole database

    arguments
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
                    
  '''
  classes = db.get_class()
  ret = {c: [] for c in classes}

  samples = sample_db_fn(db)
  for query in samples:
    ap, _ = infer(query, samples=samples)
    ret[query['cls']].append(ap)

  return ret
