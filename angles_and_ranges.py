# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:30:49 2021

@author: sjö

"""
import numpy as np
import xfab

def mod_360(angle, target):
    """
    Finds multiple of 360 to add to angle to be closest to target.
    """
    diff = angle - target
    while diff < -180:
        angle = angle + 360
        diff = angle - target
    while diff > 180:
        angle = angle - 360
        diff = angle - target
    return angle


def convert_ranges(input_ranges):
    output_ranges = []
    for r in input_ranges:
        if r[0] < 180 and r[1] > 180:
            if r[1]-360 == r[0]: output_ranges += [[-180,180]]
            else: output_ranges += [[-180,r[1]-360], [r[0],180]]
        elif r[0] > 180: output_ranges += [[r[0]-360,r[1]-360]]
        if r[1] <= 180: output_ranges += [r]
    return output_ranges


def group_to_chains(L, T): # L - list of values, T - tolerance
    # Sorts the input list of numbers (L) in ascending order, and represent them as chains (C).
    # Chain is a sequence of numbers in which the separation
    # between every two consecutive elements is smaller than tolerance (T).
    if type(L) != type([]):
        raise TypeError('Input values must be a list!')
    if T < 0:
        raise ValueError('Tolerance must not be negative!')     
    iL = np.argsort(L) # indices that would sort the elements of L in ascending order. 
    C = [] # list of chains
    C.append( [iL[0]] ) # first chain is a list consisting of just the first index in iL
    i = 0
    while i+1 < len(iL):
        if L[iL[i+1]] < L[iL[i]] + T:
            C[-1].append( iL[i+1] ) # extend the current chain
        else:
            C.append( [iL[i+1]] ) # start a new chain
        i += 1
    return C # resulting lists of chains


def merge_overlaps(ranges_in, margin, target=None):
    """ Merge all duplicating/overlapping ranges in a list of ranges. """
    if type(ranges_in) == list: # check if the input is a list
        if len(ranges_in) < 1:
            return ranges_in
        elif type(ranges_in[0]) == list: # check if its' a list of ranges
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


def disorientation(umat_1, umat_2, sgno = 141, return_axis=False):
    """
    Determines the disorientation (smallest misorientation) between grain orientations
    Input:      umat_1, umat_2 orientation matrices
                sgno: number of space group
    """
    space_group = xfab.sg.sg(sgno=sgno)
    Rs = np.unique(space_group.rot,axis=0) #unique symmetry operations in space group
    th = []
    axes = []
    for Ri in Rs:
        ui = np.dot(umat_1,Ri)
        for Rj in [np.eye(3)]:#Rs: #only need one symmetry operation I think
            uj = np.dot(umat_2,Rj)
            g = np.dot(uj,np.transpose(ui)) #(RU)^-1=(RU)^T
            detg = np.linalg.det(g)
            l = 0.5*(np.trace(g)-1)
            if np.abs(l) > 1.00000000:
                if l > 1:
                    l = 1.
                else:
                    l = -1.
            tt = np.arccos(l)
            if return_axis:
                n = (1./np.sin(tt))*(g-g.T)
                axes.append([n[2,1],-n[2,0],n[1,0]])
            th.append(tt*180./np.pi)
    if return_axis:
        return min(th), axes[np.argmin(th)]
    else:
        return min(th)