# -*- coding: utf-8 -*-

from __future__ import print_function

from DB import Database
import numpy as np
import scipy.misc
import itertools


def histogram(img, bin=4, type='global', slice=4, normalize=True):
	img = scipy.misc.imread(img, mode='RGB')
	height, width, channel = img.shape
	bins = np.linspace(0, 256, bin+1, endpoint=True)

	if type == 'global':
		hist = count_hist(img, bin, bins, channel)

	elif type == 'region':
		hist = np.zeros((slice, slice, bin ** channel))
		h_silce = np.around(np.linspace(0, height, slice+1, endpoint=True)).astype(int)
		w_slice = np.around(np.linspace(0, width, slice+1, endpoint=True)).astype(int)

		for hs in range(len(h_silce)-1):
			for ws in range(len(w_slice)-1):
				img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
				hist[hs][ws] = count_hist(img_r, bin, bins, channel)

	if normalize:
		hist /= np.sum(hist)

	return hist.flatten()

def count_hist(img, bin, bins, channel):
	bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(bin), repeat=channel))}  # permutation of bins
	hist = np.zeros(bin ** channel)

	# cluster every pixels
	for idx in range(len(bins)-1):
		img[(img >= bins[idx]) & (img < bins[idx+1])] = idx
	# add pixels into bins
	height, width, _ = img.shape
	for h in range(height):
		for w in range(width):
			b_idx = bins_idx[tuple(img[h,w])]
			hist[b_idx] += 1

	return hist


def distance():
	pass

def gen_query():
	pass

def cluster():
	pass

def inference():
	pass

def evaluate():
	pass


if __name__ == "__main__":
	db = Database()
	data = db.get_data()
	classes = db.get_class()
	print("classes:", classes)
	print("first item of class 0:", data.loc[data["class"] == classes[0]].ix[0])

	hist = histogram(data.ix[0,0], type='global')
	print("first img histogram:", hist, hist.sum())
