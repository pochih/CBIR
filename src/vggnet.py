# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG

from six.moves import cPickle
import numpy as np
import scipy.misc
import os

from evaluate import evaluate_class
from DB import Database


'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''

# configs for histogram
VGG_model  = 'vgg19'  # model type
pick_layer = 'avg'    # extract feature of this layer
d_type     = 'd1'     # distance type

depth      = 3        # retrieved depth, set to None will count the ap for whole database

''' MMAP
     depth
      depthNone, vgg19,avg,d1, MMAP 0.688624709114
      depth100,  vgg19,avg,d1, MMAP 0.754443491363
      depth30,   vgg19,avg,d1, MMAP 0.838298388513
      depth10,   vgg19,avg,d1, MMAP 0.913892057193
      depth5,    vgg19,avg,d1, MMAP 0.936158333333
      depth3,    vgg19,avg,d1, MMAP 0.941666666667
      depth1,    vgg19,avg,d1, MMAP 0.934

      (exps below use depth=None)

      vgg19,fc1,d1, MMAP 0.245548035893 (w/o subtract mean)
      vgg19,fc1,d1, MMAP 0.332583126964
      vgg19,fc1,co, MMAP 0.333836506148
      vgg19,fc2,d1, MMAP 0.294452201395
      vgg19,fc2,co, MMAP 0.297209571796
      vgg19,avg,d1, MMAP 0.688624709114
      vgg19,avg,co, MMAP 0.674217021273
'''

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)


class VGGNet(VGG):
  def __init__(self, pretrained=True, model='vgg16', requires_grad=False, remove_fc=False, show_params=False):
    super().__init__(make_layers(cfg[model]))
    self.ranges = ranges[model]
    self.fc_ranges = ((0, 2), (2, 5), (5, 7))

    if pretrained:
      exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

    if not requires_grad:
      for param in super().parameters():
        param.requires_grad = False

    if remove_fc:  # delete redundant fully-connected layer params, can save memory
      del self.classifier

    if show_params:
      for name, param in self.named_parameters():
        print(name, param.size())

  def forward(self, x):
    output = {}

    x = self.features(x)

    avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    output['avg'] = avg

    x = x.view(x.size(0), -1)  # flatten()
    dims = x.size(1)
    if dims >= 25088:
      x = x[:, :25088]
      for idx in range(len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x
    else:
      w = self.classifier[0].weight[:, :dims]
      b = self.classifier[0].bias
      x = torch.matmul(x, w.t()) + b
      x = self.classifier[1](x)
      output["fc1"] = x
      for idx in range(1, len(self.fc_ranges)):
        for layer in range(self.fc_ranges[idx][0], self.fc_ranges[idx][1]):
          x = self.classifier[layer](x)
        output["fc%d"%(idx+1)] = x

    return output


ranges = {
  'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
  'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
  'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
  'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
  'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)


class VGGNetFeat(object):

  def make_samples(self, db, verbose=True):
    sample_cache = '{}-{}'.format(VGG_model, pick_layer)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      vgg_model = VGGNet(requires_grad=False, model=VGG_model)
      vgg_model.eval()
      if use_gpu:
        vgg_model = vgg_model.cuda()
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        img = scipy.misc.imread(d_img, mode="RGB")
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= means[0]  # reduce B's mean
        img[1] -= means[1]  # reduce G's mean
        img[2] -= means[2]  # reduce R's mean
        img = np.expand_dims(img, axis=0)
        try:
          if use_gpu:
            inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
          else:
            inputs = torch.autograd.Variable(torch.from_numpy(img).float())
          d_hist = vgg_model(inputs)[pick_layer]
          d_hist = np.sum(d_hist.data.cpu().numpy(), axis=0)
          d_hist /= np.sum(d_hist)  # normalize
          samples.append({
                          'img':  d_img, 
                          'cls':  d_cls, 
                          'hist': d_hist
                         })
        except:
          pass
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


if __name__ == "__main__":
  # evaluate database
  db = Database()
  APs = evaluate_class(db, f_class=VGGNetFeat, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
