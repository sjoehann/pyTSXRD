# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:35:00 2025

@author: sjö

Class for a single surface structure
"""

import sys, os, subprocess, copy, pickle
import tifffile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy
from hexrd import imageseries
from scipy.ndimage import binary_erosion,binary_dilation
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.ndimage.interpolation import rotate
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import math
import numba


single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"

class SurfaceStructure:
    def __init__(self):
        self.log = []
        self.startp = []
        self.endp = []
        self.cord = []
        self.int = None
        self.int_range = None
        self.C = None
        return

    def add_to_log(self, str_to_add, also_print=False):
        self.log.append(str(datetime.now()) + '> ' + str_to_add)
        if also_print:
            print(str_to_add)
        return

    def set_attr(self, attr, value):
        """Method to set an attribute to the provided value and making a corresponding record in the log."""
        try:
            old = getattr(self, attr)
        except:
            old = None
        setattr(self, attr, value)
        new = getattr(self, attr)
        self.add_to_log(attr + ': ' + str(old) + ' -> ' + str(new))
        return

    def add_to_attr(self, attr, value):
        """Method to append value to the choosen attribute. The attribute must be a list."""
        try:
            old_list = getattr(self, attr)
        except:
            old_list = None
        setattr(self, attr, old_list + [value])
        new_list = getattr(self, attr)
        self.add_to_log(attr + ': += ' + str(new_list[-1]),False)
        return

    def print(self, also_log=False):
        print(double_separator + 'SurfaceAnalysis object:')
        print('start-x,start-y,start-z',self.startp)
        print('end-x,end-y,end_z',self.endp)
        if len(self.cord)>0:
            print('length of coordinates',len(self.cord[0]))
        print('Average completenes',self.C)
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log:
                print(record)
        return

    def struc_int(self, mesh):
        x_start, y_start, z_start = self.startp
        x_end, y_end, z_end = self.endp
        int_range_full = np.zeros(700)
        if z_end<=700:
            z_range = range(z_start, z_end)
        else:
            z_range = range(z_start, 700)
        x,y,z = self.cord
        int_full = mesh[x,y,z]
        int_range = [np.sum(int_full[np.where(np.array(z)==i)]) for i in z_range]
        int_range_full[z_range] = int_range
        self.set_attr("int", int_full)
        self.set_attr("int_range", int_range_full)


    

   