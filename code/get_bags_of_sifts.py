from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import pdb

def get_bags_of_sifts(image_paths, vocab_size=400):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
                                                                    
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    
    # You will want to construct SIFT features here in the same way you        #
    # did in build_vocabulary.m (except for possibly changing the sampling     #
    # rate) and then assign each local feature to its nearest cluster center   #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''

    with open('vocab{}.pkl'.format(vocab_size), 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)

    N, image_feats = len(vocab), []

    for i, path in enumerate(image_paths):
        img = np.asarray(Image.open(path),dtype='float32')
        _, descriptors = dsift(img, step=[5,5], fast=True)
        feat_dists = distance.cdist(vocab, descriptors, 'euclidean')
        feat_dist_idx = np.argmin(feat_dists, axis=0)
        hist, _ = np.histogram(feat_dist_idx, bins=np.arange(N+1))
        norm = np.linalg.norm(hist)

        if norm == 0:
            image_feats.append(hist)
        else:
            image_feats.append(hist / norm)

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
