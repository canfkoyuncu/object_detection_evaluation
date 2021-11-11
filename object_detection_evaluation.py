#!/usr/bin/env python3

"""
This script evaluates segmentation masks (binary|labeled) based on area-overlap criteria.
For any questions please contact Can Koyuncu at cfk29@case.edu

Please cite one of the following papers if you use this code:
[1] C. Koyuncu, G.N. Gunesli, et al., “DeepDistance: A Multi‐task Deep Regression Model for Cell Detection in Inverted
Microscopy Images”, Medical Image Analysis, 101720, 2020.

[2] C. Koyuncu, Rengul Cetin‐Atalay, et al., “Object oriented cell segmentation of cell nuclei in fluorescence microscopy
images”, Cytometry Part A, 2018.

[3] C. Koyuncu, E. Akhan, et al., “Iterative h‐minima based marker controlled watershed for cell nucleus segmentation”,
Cytometry Part A, 2016.

[4] C. Koyuncu, S. Arslan, et al., “Smart markers for watershed‐based cell segmentation”, PloS one, 7 (11), e48664, 2012.

"""

import numpy as np
from skimage.measure import label
from skimage.io import imread
#from matplotlib import pyplot as plt

OVERLAPPING_TH = 0.5

# The code converts a binary mask to labeled mask before evaluation.
def preprocessMask(mask):
    if mask.dtype == np.bool:
        mask = label(mask)
    elif mask.dtype != np.uint8 and mask.dtype != np.uint16:
        print(f"Warning! Mask type is different than bool|uint8|uin16 -> {mask.dtype}")
    else:
        cntLbls = np.count_nonzero(np.unique(mask))
        if cntLbls == 1:
            print(f"Warning! Only one label has been found other than background (bg label: 0). Relabeling...")
            mask = label(mask>0)

    return mask


# If at least OVERLAPPING_TH of t1 is overlapping with t2 it is a hit
def doesT1HitT2(t1, t2):
    return (np.sum(np.bitwise_and(t1,t2)) / np.sum(t1)) >= OVERLAPPING_TH


def eval(computed, gold):
    nCompLbls = np.max(computed)
    nGoldLbls = np.max(gold)

    print(f"Number of computed and groundtruth objects: {nCompLbls}, {nGoldLbls}.")

    #Decide which computed connected components hit any groundtruth ones based on overlapping criteria
    hitMatrix = np.zeros((nCompLbls, nGoldLbls), dtype=np.bool)
    for i in range(1, nCompLbls+1):
        ccComp = computed==i
        for j in range(1, nGoldLbls+1):
            if doesT1HitT2(ccComp, gold==j):
                hitMatrix[i-1, j-1] = 1
    #print(hitMatrix)

    #Decide which groundtruth connected components hit any computed ones based on overlapping criteria
    for j in range(1, nGoldLbls+1):
        ccGold = gold == j
        for i in range(1, nCompLbls+1):
            if hitMatrix[i-1, j-1] == 0 and doesT1HitT2(ccGold, computed==i):
                hitMatrix[i-1, j-1] = 1
    #print(hitMatrix)

    #Count one to one matching between computed and groundtruth CCs
    #One to one matches can be found easily by calculating Dice coefficient but with this way we could identify
    #oversegmented, undersegmented, miss, and false detections
    tp, overseg, underseg, miss, fp = 0, 0, 0, 0, 0
    for i in range(hitMatrix.shape[0]):
        hits = np.sum(hitMatrix[i,:])
        if hits == 1:
            gtComp = np.argwhere(hitMatrix[i,:]==1)[0,0]
            if np.sum(hitMatrix[:,gtComp]) == 1:
                tp += 1
        elif hits > 1:
            overseg += 1
        else:
            fp += 1

    for i in range(hitMatrix.shape[1]):
        hits = np.sum(hitMatrix[:,i])
        if hits > 1:
            underseg += 1
        elif hits == 0:
            miss += 1

    return tp, overseg, underseg, miss, fp


def calculateMetrics(tp, overseg, underseg, miss, fp):
    precision = tp / (tp+overseg+fp)
    recall = tp / (tp+underseg+miss)
    f1Score = 0 if precision==0 and recall == 0 else (2*precision*recall)/(precision+recall)
    return precision, recall, f1Score


if __name__ == "__main__":
    # ASSUMPTION: Computed and groundtruth masks are labeled from 1 to N (N: number of connected components)
    computed = imread("./computed.png")
    gold = imread("./gold.png")

    computed = preprocessMask(computed)
    gold = preprocessMask(gold)

    tp, overseg, underseg, miss, fp = eval(computed, gold)
    precision, recall, f1score = calculateMetrics(tp, overseg, underseg, miss, fp)

    print(f"TP:{tp}, Oversegmentation:{overseg}, Undersegmentation:{underseg}, Miss:{miss}, False positive:{fp}")
    print(f"Precision:{precision}, Recall:{recall}, F1-score:{f1score}")
