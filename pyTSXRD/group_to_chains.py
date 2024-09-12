import numpy as np

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