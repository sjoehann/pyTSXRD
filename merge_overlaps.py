# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:30:49 2021

@author: sjö

"""

import numpy as np
from mod_360 import mod_360

def merge_overlaps(ranges_in, margin, target=None):
    """ Merge all duplicating/overlapping ranges in a list of ranges. """
    if type(ranges_in) == list: # check if the input is a list
        if type(ranges_in[0]) == list: # check if its' a list of ranges
            pass
        elif len(ranges_in) == 2:
            ranges_in = [ranges_in]
        else:
            raise ValueError('Input range has not 2 elements!')

    if target == None:
        ranges = ranges_in
    else:   # convert each range into the [target-180, target+180] interval
        ranges =[]
        for r in ranges_in:
            if r[0] > r[1]:
                raise ValueError(f'[{r[0]}, {r[1]}] - left > right!')
            rt = [mod_360(r[0],target), mod_360(r[1],target)] # convert range
            if rt[0] > rt[1]: # if left and right limits become swaped
                ranges.append( [rt[0], target+180] ) # then this ranges breaks for two
                ranges.append( [target-180, rt[1]] )
            else: # otherwise this range is added as it is
                ranges.append( rt )

    overlapFound = True
    while overlapFound and len(ranges) > 1: # if the ranges have overaps.
        for iB in range(0,len(ranges)): # For each (Base) range,
            rB = ranges[iB]
            for iC in range(iB+1,len(ranges)): # check each (Carriage) range down in the list.
                rC = ranges[iC]
                if rB[1] < rC[0] or rC[1] < rB[0]: # If they don't overlap,
                    overlapFound = False
                    continue # then go to the next range.
                else: # If they do overlap,
                    ranges[iB] = [min(rB[0],rC[0]), max(rB[1],rC[1])] # modify this Base range,
                    ranges.remove(ranges[iC]) # and remove this Carriage range as redundant.
                    overlapFound = True
                    break # break Carriage loop as ranges was modified, and,
            if overlapFound: # break Base loop as ranges was modified.
                break
                
    ind = np.argsort([r[0] for r in ranges]) # Sort in asceding order.
    ranges = [ranges[i] for i in ind]
     
    ranges = [[r[0]+margin, r[1]-margin] for r in ranges] # cut margins
    if target != None:
        if ranges[-1][1]- ranges[0][0] < 360-2*margin: 
            pass
        else:
            ranges[0][0] = target-180
            ranges[-1][1] = target+180
    return ranges