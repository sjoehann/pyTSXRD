# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 15:34:00 2024

@author: sjö

Class to perform analysis of surface structure 
"""

import sys, os, subprocess, copy, pickle
import tifffile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyTSXRD
import scipy
from hexrd import imageseries
from scipy.ndimage import binary_erosion,binary_dilation
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.ndimage.interpolation import rotate
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import math
import numba
from scipy.ndimage import binary_dilation, binary_closing,label, find_objects,generate_binary_structure,distance_transform_edt
import logging
from collections import deque
import yaml
from tqdm import tqdm




single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"

class SurfaceAnalysis:
    def __init__(self, directory=None, name=None):
        self.log = []
        self.directory = None
        self.omega_range = None
        self.y_range = None
        self.grain_projection = None
        self.full_image = None
        self.Ind_int = None
        self.rejects = None
        self.geometry = pyTSXRD.Geometry()
        self.vmesh = None
        self.grains = []
        self.sample_pix_x = np.linspace(-3.5, 3.5, 701)
        self.sample_pix_y = np.linspace(-3.5, 3.5, 701)
        if directory:
            self.set_attr('directory', directory)
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
        print('directory:', self.directory)
        print('image directory',self.directory_images)
        print('omega range',self.omega_range)
        print('y-range',self.y_range )
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log:
                print(record)
        return
    def _process_rotation(self, args):
        g_matrix, om, sh, y_range = args
        mat = rotate(g_matrix, om, axes=(1, 0), reshape=False, order=1, mode='constant', cval=0)
        mat[mat < 0.01] = 0
        mat = np.sum(mat, axis=0)

        nz = np.nonzero(mat)[0]

        min_g = np.argmin(np.abs(y_range - (sh - np.max(nz)) / 100))
        max_g = np.argmin(np.abs(y_range - (sh - np.min(nz)) / 100))
        return min_g, max_g

    def prep_projections(self):
        """Prepares rotated matrix of the grain"""
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm

        N_grains = len(self.grains)
        num_images = len(self.omega_range)
        grain_extents = np.zeros((N_grains, num_images, 2), dtype=np.int32)
        sh = np.shape(self.grains[0].matrix)[0] // 2
        y_range = self.y_range[1:-1]

        tasks = []
        for i, om in enumerate(self.omega_range):
            for g in self.grains:
                tasks.append((g.matrix, om, sh, y_range))

        print("Starting parallel rotation..")
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._process_rotation, tasks), total=len(tasks)))

        idx = 0
        for i in range(num_images):
            for n in range(N_grains):
                grain_extents[n, i, :] = results[idx]
                idx += 1

        self.set_attr("grain_projection", grain_extents)
        print("All grains prepared")
        return


    def process_images(self,block=10,threshold=0.5,size_th=10,binning=1,image_rows=None,eta_ranges=None,min_overlap=[0.8,0.8,0.6],remove_background=True,min_peak_dist=4,min_support=4):
        '''Load images and find signals'''
        if self.imProcessor is None:
            raise ValueError('No ImProcessor has been assigned.')

        self.imProcessor.process_images(block=block,threshold=threshold,size_th=size_th,binning=binning,image_rows=image_rows,
                                        eta_ranges=eta_ranges,min_overlap=min_overlap,min_peak_dist=min_peak_dist,min_support=min_support)

        if remove_background:
            self.imProcessor.set_attr('background',None)


    def merge_signals(self, size_th, min_overlap=[0.8, 0.8, 0.6]):
        '''Merge signals before sorting'''
        ids_all = self.full_image[0].astype(np.int64, copy=False)
        ints_all = self.full_image[1]
        xs_all = self.full_image[2].astype(np.int64, copy=False)
        ys_all = self.full_image[3].astype(np.int64, copy=False)
        dys_all = self.full_image[4].astype(np.int64, copy=False)
        oms_all = self.full_image[5].astype(np.int64, copy=False)

        if ids_all.size == 0:
            return

        order = np.argsort(ids_all, kind="mergesort")
        ids_sorted = ids_all[order]
        cuts = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1], True])
        starts = cuts[:-1]
        ends = cuts[1:]
        unique_ids = ids_sorted[starts]

        id_to_idx = {}
        for k, uid in enumerate(unique_ids):
            id_to_idx[uid] = order[starts[k]:ends[k]]

        signal_groups = []
        for uid in unique_ids:
            idx = id_to_idx[uid]

            pixels_by_omega_dy = {}
            pixels_by_omega = {}
            dys_by_omega = {}
            omegas = set()

            for i in idx:
                om = int(oms_all[i])
                dy = int(dys_all[i])
                x = int(xs_all[i])
                y = int(ys_all[i])

                omegas.add(om)

                key = (om, dy)
                if key not in pixels_by_omega_dy:
                    pixels_by_omega_dy[key] = set()
                pixels_by_omega_dy[key].add((x, y))

                if om not in pixels_by_omega:
                    pixels_by_omega[om] = set()
                pixels_by_omega[om].add((x, y))

                if om not in dys_by_omega:
                    dys_by_omega[om] = set()
                dys_by_omega[om].add(dy)

            signal_groups.append({
                "id": int(uid),
                "indices": idx.tolist(),
                "pixels_by_omega_dy": pixels_by_omega_dy,
                "pixels_by_omega": pixels_by_omega,
                "dys_by_omega": dys_by_omega,
                "omegas": omegas})

        def can_merge(g1, g2):
            for om1 in g1["omegas"]:
                for om2 in g2["omegas"]:
                    if om1 == om2 or abs(om1 - om2) > 1:
                        continue

                    dys1 = g1["dys_by_omega"].get(om1, set())
                    dys2 = g2["dys_by_omega"].get(om2, set())
                    shared_dys = dys1 & dys2

                    if not shared_dys:
                        continue

                    dy_frac_1 = len(shared_dys) / len(dys1) if len(dys1) > 0 else 0
                    dy_frac_2 = len(shared_dys) / len(dys2) if len(dys2) > 0 else 0

                    if dy_frac_1 < min_overlap[1] and dy_frac_2 < min_overlap[1]:
                        continue

                    overlap_sum = 0
                    total_pix_1 = 0
                    total_pix_2 = 0

                    for dy in shared_dys:
                        p1 = g1["pixels_by_omega_dy"].get((om1, dy), set())
                        p2 = g2["pixels_by_omega_dy"].get((om2, dy), set())

                        overlap_sum += len(p1 & p2)
                        total_pix_1 += len(p1)
                        total_pix_2 += len(p2)

                    if total_pix_1 == 0 or total_pix_2 == 0:
                        continue

                    pix_frac_1 = overlap_sum / total_pix_1
                    pix_frac_2 = overlap_sum / total_pix_2

                    if pix_frac_1 >= min_overlap[2] or pix_frac_2 >= min_overlap[2]:
                        return True

            return False

        merged_groups = []
        used = set()

        for i, grp in enumerate(tqdm(signal_groups, desc="Merging signals")):
            if i in used:
                continue

            current = {
                "ids": {grp["id"]},
                "indices": list(grp["indices"]),
                "pixels_by_omega_dy": {k: set(v) for k, v in grp["pixels_by_omega_dy"].items()},
                "pixels_by_omega": {k: set(v) for k, v in grp["pixels_by_omega"].items()},
                "dys_by_omega": {k: set(v) for k, v in grp["dys_by_omega"].items()},
                "omegas": set(grp["omegas"])
            }

            changed = True
            used.add(i)

            while changed:
                changed = False
                for j, grp2 in enumerate(signal_groups):
                    if j in used:
                        continue

                    test_group = {
                        "id": grp2["id"],
                        "indices": grp2["indices"],
                        "pixels_by_omega_dy": grp2["pixels_by_omega_dy"],
                        "pixels_by_omega": grp2["pixels_by_omega"],
                        "dys_by_omega": grp2["dys_by_omega"],
                        "omegas": grp2["omegas"]
                    }

                    if not can_merge(current, test_group):
                        continue

                    current["ids"].add(grp2["id"])
                    current["indices"].extend(grp2["indices"])
                    current["omegas"].update(grp2["omegas"])

                    for key, val in grp2["pixels_by_omega_dy"].items():
                        if key not in current["pixels_by_omega_dy"]:
                            current["pixels_by_omega_dy"][key] = set()
                        current["pixels_by_omega_dy"][key].update(val)

                    for key, val in grp2["pixels_by_omega"].items():
                        if key not in current["pixels_by_omega"]:
                            current["pixels_by_omega"][key] = set()
                        current["pixels_by_omega"][key].update(val)

                    for key, val in grp2["dys_by_omega"].items():
                        if key not in current["dys_by_omega"]:
                            current["dys_by_omega"][key] = set()
                        current["dys_by_omega"][key].update(val)

                    used.add(j)
                    changed = True

            merged_groups.append(current)

        new_signals = [[], [], [], [], [], []]
        new_index = 0

        for grp in merged_groups:
            idx = grp["indices"]

            om_vals = oms_all[idx]
            dy_vals = dys_all[idx]
            x_vals = xs_all[idx]
            y_vals = ys_all[idx]

            if dy_vals.max() == dy_vals.min():
                continue

            if (np.max(x_vals) - np.min(x_vals)) * (np.max(y_vals) - np.min(y_vals)) < size_th:
                continue

            for i in idx:
                new_signals[0].append(new_index)
                new_signals[1].append(ints_all[i])
                new_signals[2].append(xs_all[i])
                new_signals[3].append(ys_all[i])
                new_signals[4].append(dys_all[i])
                new_signals[5].append(oms_all[i])

            new_index += 1

        self.set_attr("full_image", np.array(new_signals))



    def sort_signals(self, sort_th, merge=True ,min_overlap=[0.8, 0.8, 0.6],size_th=10,beta=0.1): 
        '''Asign signals to grains'''
        if merge: 
            print(f"Berfore merging: {len(np.unique(self.full_image[0]))} signals. Merging signals... ") 
            self.merge_signals(size_th=size_th,min_overlap=min_overlap) 
            print(f"After merging: {len(np.unique(self.full_image[0]))} signals.") 
        print("Assigning signals to grains") 

        N_grains = len(self.grains) 
        n_images = len(self.omega_range) 
        Signals_sorted = [[] for _ in range(N_grains)] 
        rejects = [] 
        grain_projection = np.array(self.grain_projection) 
        all_ids = np.array(self.full_image[0]) 
        unique_signals = np.unique(all_ids) 
        dys_all = np.array(self.full_image[4]) 
        oms_all = np.array(self.full_image[5]) 
        for ns in unique_signals: 
            si = np.where(all_ids == ns) 
            dys = dys_all[si] 
            if np.max(dys)==np.min(dys):
                continue
            oms = oms_all[si] 
            unique_oms = np.unique(oms)
            best_span = -1 
            best_o = None 
            for o in np.unique(oms): 
                omi = (oms == o) 
                dy_span = dys[omi].max() - dys[omi].min() 
                if dy_span > best_span: 
                    best_span = dy_span 
                    best_o = o 
            omi = (oms == best_o)
            dy_min = dys[omi].min() 
            dy_max = dys[omi].max() 
            dy_c = round(np.mean([dy_min,dy_max]))

            best_grain = []
            best_score = np.inf

            for gn in range(N_grains):
                diff_l = []
                max_l = []

                y1 = grain_projection[gn, int(best_o), 0]
                y2 = grain_projection[gn, int(best_o), 1]
                yc = 0.5 * (y1 + y2)

                omi = oms == best_o
                dy_min = dys[omi].min()
                dy_max = dys[omi].max()
                dy_c = 0.5 * (dy_min + dy_max)

                d1 = beta[0] * abs(dy_min - y1)
                d2 = beta[1] * abs(dy_max - y2)
                dc = beta[2] * abs(dy_c - yc)

                diff = d1 + d2 + dc
                diff_m = max(d1, d2)

                if y1 == 0 or y2 == 70:
                    diff -= 1
                    diff_m -= 1

                if not (diff <= sort_th and diff_m <= sort_th / 3 + 1 and dy_max > y1 and dy_min < y2):
                    continue

                diff_l.append(diff)
                max_l.append(diff_m)

                for o in unique_oms:
                    omi = oms == o
                    dy_min = dys[omi].min()
                    dy_max = dys[omi].max()
                    dy_c = 0.5 * (dy_min + dy_max)
                    dy_span2 = dy_max - dy_min

                    if best_span - dy_span2 > 1:
                        continue

                    y1 = grain_projection[gn, int(o), 0]
                    y2 = grain_projection[gn, int(o), 1]
                    yc = 0.5 * (y1 + y2)

                    d1 = beta[0] * abs(dy_min - y1)
                    d2 = beta[1] * abs(dy_max - y2)
                    dc = beta[2] * abs(dy_c - yc)

                    diff_2 = d1 + d2 + dc
                    diff_2_m = max(d1, d2)

                    if y1 == 0 or y2 == 70:
                        diff_2 -= 1
                        diff_2_m -= 1

                    diff_l.append(diff_2)
                    max_l.append(diff_2_m)

                mean_diff = np.mean(diff_l)
                max_diff = np.max(max_l)

                score = mean_diff + 2.0 * max_diff

                if mean_diff < sort_th and max_diff <= sort_th / 3 + 1:
                    if score < best_score:
                        best_score = score
                        best_grain = [gn]
                    elif score == best_score:
                        best_grain.append(gn)
            if len(best_grain)==0:
                rejects.append(ns)
            else:
                for bg in best_grain: 
                    Signals_sorted[bg].append(ns) 

        self.set_attr("allgrains", Signals_sorted) 
        self.set_attr("rejects",rejects)

    def plot_sinogram(self,grain,color="green",y_axis='auto',x_axis='auto',x_label='auto',y_label='auto',labelsize=15,ticksize=12,dpi=200,with_fit=True,compare=False,savefig=False,filename='sinogram.png'):
        """Plot grain projections and assigned diffraction signals."""

        if grain == 'all':
            pl_range = range(len(self.grains))
        elif isinstance(grain,list):
            pl_range = grain
        elif isinstance(grain,int):
            pl_range = [grain]
        else:
            print('Grain must be "all", a list, or an integer')
            return

        if x_axis == 'auto':
            x_ticks = np.linspace(np.min(self.omega_range),np.max(self.omega_range),5)
        elif isinstance(x_axis,list):
            x_ticks = x_axis
        else:
            print('x_axis must be "auto" or a list')
            return

        if y_axis == 'auto':
            y_ticks = np.linspace(np.min(self.y_range),np.max(self.y_range),5)
        elif isinstance(y_axis,list):
            y_ticks = y_axis
        else:
            print('y_axis must be "auto" or a list')
            return

        if x_label == 'auto':
            x_label = r'$\omega$ (deg.)'

        if y_label == 'auto':
            y_label = 'y (mm)'

        indtolen = len(self.sample_pix_y-1)/(len(self.y_range)-3)
        sample_half = self.y_range[-2]

        grain_extents = self.grain_projection
        signal_data = np.array(self.full_image)
        Signals_sorted = self.allgrains

        for gn in pl_range:
            print(f'Grain number {gn}')

            y1 = np.array([grain_extents[gn,i,0] for i in range(len(self.omega_range))])/indtolen-sample_half
            y2 = np.array([grain_extents[gn,i,1] for i in range(len(self.omega_range))])/indtolen-sample_half

            plt.figure(figsize=(5,5),dpi=dpi)

            if with_fit:
                print(f'Number of signals: {len(Signals_sorted[gn])}')
                for sg in Signals_sorted[gn]:
                    inds_sg = np.where(signal_data[0] == sg)[0]
                    om_plot = [self.omega_range[int(signal_data[5][i])] for i in inds_sg]
                    plt.plot(om_plot,signal_data[4][inds_sg]/indtolen-sample_half,marker='s',ms=1)

            plt.plot(self.omega_range,y1,'-',c=color)
            plt.plot(self.omega_range,y2,'-',c=color)

            if compare:
                if not isinstance(compare,int):
                    print('Compare must be an integer (grain number)')
                else:
                    y1_compare = np.array([grain_extents[compare,i,0] for i in range(len(self.omega_range))])/indtolen-sample_half
                    y2_compare = np.array([grain_extents[compare,i,1] for i in range(len(self.omega_range))])/indtolen-sample_half
                    plt.plot(self.omega_range,y1_compare,'-',c='red')
                    plt.plot(self.omega_range,y2_compare,'-',c='red')

            plt.xticks(x_ticks,fontsize=ticksize)
            plt.yticks(y_ticks,fontsize=ticksize)
            plt.xlabel(x_label,fontsize=labelsize)
            plt.ylabel(y_label,fontsize=labelsize)


            if savefig:
                plt.savefig(self.directory+filename,transparent=True,bbox_inches='tight')

            plt.show()

        return


    def plot_det(self,grain,binning,plot_type="image",y_axis='auto',y_lim='auto',x_axis='auto',x_lim='auto',x_label='auto',y_label='auto',
                 labelsize=15,ticksize=12,dpi=200,savefig=False,filename='im.png',exp_bragg=False,meas_bragg=False,out_bragg=False,limits=False,disc_th=None,omega_range=None,y_range=None):

        if grain == 'all':
            pl_range = range(len(self.grains))
        elif isinstance(grain,list):
            pl_range = grain
        elif isinstance(grain,int):
            pl_range = [grain]
        else:
            print('Grain must be "all", a list, or an integer')
            return

        det_x = self.geometry.dety_size//binning
        det_y = self.geometry.detz_size//binning

        if x_axis == 'auto':
            x_ticks = []
        elif isinstance(x_axis,list):
            x_ticks = x_axis
        else:
            print('x_axis must be "auto" or a list')
            return

        if y_axis == 'auto':
            y_ticks = []
        elif isinstance(y_axis,list):
            y_ticks = y_axis
        else:
            print('y_axis must be "auto" or a list')
            return

        if x_lim == 'auto':
            x_lim = [0,det_x]
        elif not isinstance(x_lim,list):
            print('x_lim must be "auto" or a list')
            return

        if y_lim == 'auto':
            y_lim = [0,det_y]
        elif not isinstance(y_lim,list):
            print('y_lim must be "auto" or a list')
            return

        if x_label == 'auto':
            x_label = None

        if y_label == 'auto':
            y_label = None

        if omega_range is not None and (not isinstance(omega_range,list) or len(omega_range) != 2):
            print('omega_range must be None or [min,max]')
            return

        if y_range is not None and (not isinstance(y_range,list) or len(y_range) != 2):
            print('y_range must be None or [min,max]')
            return

        Signals_sorted = self.allgrains
        signal_data = np.asarray(self.full_image)

        for g in pl_range:
            print(f'Grain number {g}')

            n_meas = len(self.grains[g].measured_gvectors)
            n_exp = len(self.grains[g].expected_gvectors)

            xc = np.array([det_x-self.grains[g].measured_gvectors[i]["xc"]//binning for i in range(n_meas)])
            yc = np.array([self.grains[g].measured_gvectors[i]["yc"]//binning for i in range(n_meas)])

            xce = np.array([det_x-self.grains[g].expected_gvectors[i]["xc"]//binning for i in range(n_exp)])
            yce = np.array([self.grains[g].expected_gvectors[i]["yc"]//binning for i in range(n_exp)])

            plt.figure(figsize=(8,4),dpi=dpi)

            if plot_type == "image":
                image = np.zeros((det_y,det_x))

                for sg in Signals_sorted[g]:
                    inds = np.where(signal_data[0] == sg)[0]

                    if omega_range is not None:
                        inds = inds[(signal_data[5,inds] >= omega_range[0]) & (signal_data[5,inds] <= omega_range[1])]

                    if y_range is not None:
                        inds = inds[(signal_data[4,inds] >= y_range[0]) & (signal_data[4,inds] <= y_range[1])]

                    intensity = signal_data[1,inds]
                    x = signal_data[2,inds].astype(int)
                    y = signal_data[3,inds].astype(int)

                    valid = (x >= 0) & (x < det_x) & (y >= 0) & (y < det_y)
                    np.maximum.at(image,(y[valid],x[valid]),intensity[valid])

                plt.imshow(image,vmax=900,cmap='magma',origin='upper')

            elif plot_type == "signals":
                for sg in Signals_sorted[g]:
                    inds = np.where(signal_data[0] == sg)[0]

                    if omega_range is not None:
                        inds = inds[(signal_data[5,inds] >= omega_range[0]) & (signal_data[5,inds] <= omega_range[1])]

                    if y_range is not None:
                        inds = inds[(signal_data[4,inds] >= y_range[0]) & (signal_data[4,inds] <= y_range[1])]

                    plt.plot(signal_data[2,inds],signal_data[3,inds],'.',ms=1)

            else:
                print('plot_type must be "image" or "signals"')
                return

            if limits:
                if disc_th is None:
                    print('disc_th must be given when limits=True')
                else:
                    xsq1 = self.geometry.y_center/binning-disc_th[0]
                    xsq2 = self.geometry.y_center/binning+disc_th[0]
                    plt.plot([xsq1,xsq1,xsq2,xsq2,xsq1],[disc_th[1][0],disc_th[1][1],disc_th[1][1],disc_th[1][0],disc_th[1][0]],'w--',alpha=0.5)

            if exp_bragg:
                good_q = self.grains[g].sort_qs[0][1]
                bad_q = self.grains[g].sort_qs[1][1]
                plt.plot(xce[good_q],yce[good_q],"g+")
                plt.plot(xce[bad_q],yce[bad_q],"r+")

            if meas_bragg:
                good_q = self.grains[g].sort_qs[0][0]
                bad_q = self.grains[g].sort_qs[1][0]
                plt.plot(xc[good_q],yc[good_q],"g+")
                plt.plot(xc[bad_q],yc[bad_q],"r+")

            if out_bragg:
                out_q = self.grains[g].sort_qs[2][1]
                plt.plot(xce[out_q],yce[out_q],"b+")

            plt.xticks(x_ticks,fontsize=ticksize)
            plt.yticks(y_ticks,fontsize=ticksize)
            plt.xlabel(x_label,fontsize=labelsize)
            plt.ylabel(y_label,fontsize=labelsize)
            plt.xlim(x_lim)
            plt.ylim(y_lim)

            if savefig:
                plt.savefig(self.directory+str(g)+filename,transparent=True,bbox_inches='tight')

            plt.show()




    def calculate_reciprocal_mesh(self,grainnumber,binning,angi,mesh=True,bragg=False):
        '''Calculate resiprocal coordinates for signals'''
        det_size = [self.geometry.detz_size//binning, (self.geometry.dety_size//binning)]
        q0_pos = [(self.geometry.y_center)/binning, self.geometry.z_center/binning]
        pix_size = self.geometry.y_size*1e-6*binning
        SDD = self.geometry.distance*1e-6
        lam = self.geometry.wavelength
        k0 = 2*np.pi/(lam)
        deltalab = np.zeros((det_size[0],det_size[1],4))
        num_images = len(self.omega_range)
        print("Calculating lab frame     ", end="\r")
        for i in range(det_size[0]):
            for j in range (det_size[1]):
                deltalab[i,j,:] = pix2lab(q0_pos[0],q0_pos[1],i,j,pix_size,SDD,k0)
        deltaK = np.zeros((det_size[0],det_size[1],num_images,3))
        om_range_rad = np.deg2rad(self.omega_range)
        rotation_matrices = np.array([rotmat(np.deg2rad(angi), om) for om in om_range_rad])
        print("Calculating rotated frame       ", end="\r")
        for o in range(num_images):
            M = rotation_matrices[o]  
            deltaK[:, :, o, :] = np.einsum('ij,xyj->xyi', M, deltalab[:,:,:3])
        print("Creating mesh                  ", end="\r")
        binsize = np.array([0.01,0.01,0.01])
        hmax = round(np.max(deltaK[:,:,:,0])/binsize[0])*binsize[0]
        hmin = round(np.min(deltaK[:,:,:,0])/binsize[0])*binsize[0]
        kmax = round(np.max(deltaK[:,:,:,1])/binsize[1])*binsize[1]
        kmin = round(np.min(deltaK[:,:,:,1])/binsize[1])*binsize[1]
        lmax = round(np.max(deltaK[:,:,:,2])/binsize[2])*binsize[2]
        lmin = round(np.min(deltaK[:,:,:,2])/binsize[2])*binsize[2]
        h = np.linspace(hmin,hmax,int((hmax-hmin)/binsize[0])+1)
        k = np.linspace(kmin,kmax,int((kmax-kmin)/binsize[1])+1)
        l = np.linspace(lmin,lmax,int((lmax-lmin)/binsize[2])+1)
        if mesh:
            vmesh = compute_vmesh(h, k, l, deltaK, self.Ind_int)
            self.set_attr("vmesh",vmesh)
            print("Reciprocal map done!                 ", end="\r")
        if bragg:
            n = grainnumber
            xc = [int(self.grains[n].measured_gvectors[i]["xc"]//binning) for i in range(len(self.grains[n].measured_gvectors))]
            yc = [int(self.grains[n].measured_gvectors[i]["yc"]//binning) for i in range(len(self.grains[n].measured_gvectors))]
            oc = [int(np.argmin(abs(self.omega_range+self.grains[n].measured_gvectors[i]["omega"]))) for i in range(len(self.grains[n].measured_gvectors))]
            for i in range(len(self.grains[n].expected_gvectors)):
                xc.append(int(self.grains[n].expected_gvectors[i]["xc"]//binning))
                yc.append(int(self.grains[n].expected_gvectors[i]["yc"]//binning))
                oc.append(int(np.argmin(abs(self.omega_range+self.grains[n].expected_gvectors[i]["omega"]))))
            bragg_spots = [oc,xc,yc]
            Q = compute_vmesh(h, k, l, deltaK, self.Ind_int,spots=bragg_spots,bragg=True)
            print("Reciprocal braggspots done!                 ", end="\r")
            return h,k,l,Q
        return h,k,l	

    def sort_surfstruc(self,g,binning,dist_th,disc_th,Q):
        """Evaluate surface signals compared to the Bragg spots of the different grains"""

        shx,shy,shz = self.vmesh.shape
        xy_th,z_th = dist_th

        q0_pos = [self.geometry.y_center/binning,self.geometry.z_center/binning]
        x_min,x_max = q0_pos[0]-disc_th[0],q0_pos[0]+disc_th[0]
        y_min,y_max = disc_th[1]

        n_meas = len(self.grains[g].measured_gvectors)
        n_exp = len(self.grains[g].expected_gvectors)
        n_q = n_meas+n_exp

        qx = Q[:,0]
        qy = shy-Q[:,1]
        qz = Q[:,2]

        xc = np.array([self.geometry.dety_size//binning-self.grains[g].measured_gvectors[i]["xc"]//binning for i in range(n_meas)])
        yc = np.array([self.grains[g].measured_gvectors[i]["yc"]//binning for i in range(n_meas)])
        xce = np.array([self.geometry.dety_size//binning-self.grains[g].expected_gvectors[i]["xc"]//binning for i in range(n_exp)])
        yce = np.array([self.grains[g].expected_gvectors[i]["yc"]//binning for i in range(n_exp)])

        om_min = np.min(self.omega_range)
        om_max = np.max(self.omega_range)

        oc = np.array([-self.grains[g].measured_gvectors[i]["omega"] for i in range(n_meas)])
        oce = np.array([-self.grains[g].expected_gvectors[i]["omega"] for i in range(n_exp)])

        valid_meas = np.where((oc >= om_min) & (oc <= om_max))[0]
        valid_exp = np.where((oce >= om_min) & (oce <= om_max))[0]

        valid_meas_set = set(valid_meas)
        valid_exp_set = set(valid_exp)

        valid_global = set(valid_meas)
        valid_global.update(n_meas+i for i in valid_exp)

        q_outside = [[],[]]

        for i in valid_meas:
            if xc[i] < x_min or xc[i] > x_max or yc[i] < y_min or yc[i] > y_max:
                q_outside[0].append(i)

        for i in valid_exp:
            if xce[i] < x_min or xce[i] > x_max or yce[i] < y_min or yce[i] > y_max:
                q_outside[1].append(i)

        outside_global = set(q_outside[0])
        outside_global.update(n_meas+i for i in q_outside[1])

        good_sc = []
        bad_sc = []
        outside_sc = []
        good_q_global = set()

        for i,sc in enumerate(self.grains[g].surfstrucs):
            x,y,z = sc.cord
            xm = np.mean(x)
            ym = np.mean(y)
            z = np.asarray(z)

            xyd = np.sqrt((qx-xm)**2+(qy-ym)**2)
            candidates = np.where(xyd < xy_th)[0]
            candidates = np.array([j for j in candidates if j in valid_global])

            if len(candidates) == 0:
                bad_sc.append(i)
                continue

            z_dist = np.min(np.abs(z[:,None]-qz[candidates][None,:]),axis=0)
            candidates = candidates[z_dist < z_th]

            if len(candidates) == 0:
                bad_sc.append(i)
                continue

            good_sc.append(i)
            good_q_global.update(candidates.tolist())

            closest = candidates[np.argmin(xyd[candidates])]

            if closest in outside_global:
                outside_sc.append(i)

        good_q = [[],[]]
        bad_q = [[],[]]

        for i in valid_meas:
            if i in good_q_global:
                good_q[0].append(i)
            else:
                bad_q[0].append(i)

        for i in valid_exp:
            global_i = n_meas+i

            if global_i in good_q_global:
                good_q[1].append(i)
            else:
                bad_q[1].append(i)

        self.grains[g].set_attr("sort_ind",[good_sc,bad_sc,outside_sc])
        self.grains[g].set_attr("sort_qs",[good_q,bad_q,q_outside])

        good_q_set = set(good_q[1])-set(q_outside[1])
        bad_q_set = set(bad_q[1])-set(q_outside[1])

        good_sc_set = set(good_sc)-set(outside_sc)
        bad_sc_set = set(bad_sc)-set(outside_sc)

        goodQ_len = len(good_q_set)
        badQ_len = len(bad_q_set)
        goodSC_len = len(good_sc_set)
        badSC_len = len(bad_sc_set)

        if goodQ_len+badQ_len == 0 or goodSC_len+badSC_len == 0:
            comp = 0
            acc = 0
        else:
            comp = goodQ_len/(goodQ_len+badQ_len)
            acc = goodSC_len/(goodSC_len+badSC_len)

        self.grains[g].set_attr("acc",acc)
        self.grains[g].set_attr("comp",comp)

        print(f'Grain {g}:')
        print(f'Bragg spots {goodQ_len+badQ_len}, completeness {round(comp,2)}')
        print(f'Signals {goodSC_len+badSC_len}, accuracy {round(acc,2)}')


    def analyse_CTRs(self,binning,angi,th_groupsize,tolerance,th_overlap,sigtoq_distth,disc_th,grain='all',plot=True):
        """Evaluate surface signals compared to the Bragg spots of the different grains"""
        if grain == 'all':
            pl_range = range(len(self.grains))
        elif isinstance(grain,list):
            pl_range = grain
        elif isinstance(grain,int):
            pl_range = [grain]
        else:
            print('Grain must be "all", a list, or an integer')
            return

        signal_data = np.asarray(self.full_image)
        n_images = len(self.omega_range)
        det_z = self.geometry.detz_size//binning
        det_y = self.geometry.dety_size//binning

        for g in pl_range:
            print(f'Grain number {g}')

            image = np.zeros((n_images,det_z,det_y))

            for sg in self.allgrains[g]:
                inds = np.where(signal_data[0] == sg)[0]

                intensity = signal_data[1,inds]
                x = signal_data[2,inds].astype(int)
                y = signal_data[3,inds].astype(int)
                o = signal_data[5,inds].astype(int)

                valid = (o >= 0) & (o < n_images) & (y >= 0) & (y < det_z) & (x >= 0) & (x < det_y)
                np.maximum.at(image,(o[valid],y[valid],x[valid]),intensity[valid])

            nz_index = np.nonzero(image)
            nz_c = image[nz_index]
            self.set_attr("Ind_int",[nz_index,nz_c])

            h,k,l,Q = self.calculate_reciprocal_mesh(g,binning,angi,mesh=True,bragg=True)

            nz_pixels = np.argwhere(self.vmesh > 0)
            shx,shy,shz = self.vmesh.shape

            selected_groups = []
            used_coordinates = set()

            print("Selecting groups over threshold")

            for cord in nz_pixels:
                key = tuple(cord)

                if key in used_coordinates:
                    continue

                group_coords = find_connected_component(self.vmesh,cord,2*tolerance)

                for c in group_coords:
                    used_coordinates.add(tuple(c))

                if len(group_coords) >= th_groupsize:
                    selected_groups.append(np.asarray(group_coords))

            print(f"Before combining: {len(selected_groups)} groups")

            combined_groups = []

            while selected_groups:
                current = selected_groups.pop(0)
                changed = True

                while changed:
                    changed = False
                    current_xy = set(map(tuple,current[:,:2]))
                    remaining = []

                    for other in selected_groups:
                        other_xy = set(map(tuple,other[:,:2]))
                        overlap = len(current_xy & other_xy)

                        if overlap > th_overlap*len(current_xy) and overlap > th_overlap*len(other_xy):
                            current = np.vstack((current,other))
                            current_xy = set(map(tuple,current[:,:2]))
                            changed = True
                        else:
                            remaining.append(other)

                    selected_groups = remaining

                combined_groups.append(current)

            print(f"After combining: {len(combined_groups)} groups")

            self.grains[g].surfstrucs = []

            for gr in combined_groups:
                SC = pyTSXRD.SurfaceStructure()

                x = gr[:,0].tolist()
                y = gr[:,1].tolist()
                z = gr[:,2].tolist()

                SC.set_attr('cord',[x,y,z])

                min_ind = np.argmin(z)
                max_ind = np.argmax(z)

                x_min,x_max = x[min_ind],x[max_ind]
                y_min,y_max = y[min_ind],y[max_ind]
                z_min,z_max = z[min_ind],z[max_ind]

                if abs(x_min-x_max) <= 10 and abs(y_min-y_max) <= 10:
                    x_av = round((x_min+x_max)/2)
                    y_av = round((y_min+y_max)/2)

                    SC.set_attr('startp',[x_av,y_av,z_min])
                    SC.set_attr('endp',[x_av,y_av,z_max])

                elif z_max != z_min:
                    x0 = x_min+(x_max-x_min)/(z_max-z_min)*(shz/2-z_min)
                    x1 = x_min+(x_max-x_min)/(z_max-z_min)*(shz-z_min)
                    y0 = y_min+(y_max-y_min)/(z_max-z_min)*(shz/2-z_min)
                    y1 = y_min+(y_max-y_min)/(z_max-z_min)*(shz-z_min)

                    SC.set_attr('startp',[round(x0),round(y0),z_min])
                    SC.set_attr('endp',[round(x1),round(y1),z_max])

                else:
                    SC.set_attr('startp',[round(np.mean(x)),round(np.mean(y)),z_min])
                    SC.set_attr('endp',[round(np.mean(x)),round(np.mean(y)),z_max])

                SC.struc_int(self.vmesh)
                self.grains[g].surfstrucs.append(SC)

            self.sort_surfstruc(g,binning,sigtoq_distth,disc_th,Q)

        return

    def plot_surf(self,plot_type='orientation',also_save=False,save_name='ball.png',y_axis='auto',x_axis='auto',x_label='auto',y_label='auto',
              labelsize=15,ticksize=12,marksize=12,mark_grainnumber=False,mark_millers=False,show_pos=False,dpi=150,num_print=False,num_cut=0.6):
        """Plot surface grain map."""

        if x_axis == 'auto':
            x_ticks = np.linspace(np.min(self.y_range),np.max(self.y_range),5)
        elif isinstance(x_axis,list):
            x_ticks = x_axis
        else:
            print('x_axis must be "auto" or a list')
            return

        if y_axis == 'auto':
            y_ticks = np.linspace(np.min(self.y_range),np.max(self.y_range),5)
        elif isinstance(y_axis,list):
            y_ticks = y_axis
        else:
            print('y_axis must be "auto" or a list')
            return

        if x_label == 'auto':
            x_label = 'x (mm)'

        if y_label == 'auto':
            y_label = 'y (mm)'

        if plot_type == 'orientation':
            euler_list = [g.phi for g in self.grains]
            plt.rcParams["axes.grid"] = False

            ori = Orientation.from_euler(np.deg2rad(euler_list))
            ipfkey = plot.IPFColorKeyTSL(symmetry.Oh)
            ori.symmetry = ipfkey.symmetry
            color_matrix = ipfkey.orientation2color(ori)

            ori.scatter("ipf",c=color_matrix,direction=ipfkey.direction)
            plt.title('')

            if also_save:
                plt.savefig(self.directory+'triangle_'+save_name,facecolor='white',bbox_inches='tight',transparent=True)

            figure,ax = plt.subplots(figsize=(7,7),dpi=dpi)

            for i,g in enumerate(self.grains):
                x_mm = g.index[0]/(len(self.sample_pix_x)-1)*(self.y_range[-2]-self.y_range[1])+self.y_range[1]
                y_mm = g.index[1]/(len(self.sample_pix_y)-1)*(self.y_range[-2]-self.y_range[1])+self.y_range[1]

                plt.scatter(x_mm,y_mm,color=color_matrix[i],marker='s',s=1,rasterized=True)

                xm = np.mean(x_mm)
                ym = np.mean(y_mm)

                if mark_grainnumber:
                    plt.annotate(f'({i})',(xm,ym),fontsize=marksize)

                if mark_millers:
                    plt.annotate(f'({int(g.miller[0])},{int(g.miller[1])},{int(g.miller[2])})',(xm,ym),fontsize=marksize)

                if show_pos:
                    plt.scatter(xm,ym,c='k')

            plt.xticks(x_ticks,fontsize=ticksize)
            plt.yticks(y_ticks,fontsize=ticksize)
            plt.xlabel(x_label,fontsize=labelsize)
            plt.ylabel(y_label,fontsize=labelsize)
            plt.axis('equal')

            if also_save:
                plt.savefig(self.directory+save_name,transparent=True,bbox_inches='tight')

            plt.show()

        elif plot_type in ['comp','acc']:
            figure,ax = plt.subplots(figsize=(7,7),dpi=dpi)

            for i,g in enumerate(self.grains):
                value = g.comp if plot_type == 'comp' else g.acc



                x_mm = g.index[0]/(len(self.sample_pix_x)-1)*(self.y_range[-2]-self.y_range[1])+self.y_range[1]
                y_mm = g.index[1]/(len(self.sample_pix_y)-1)*(self.y_range[-2]-self.y_range[1])+self.y_range[1]

                values = np.full(len(x_mm),value)
                sc = plt.scatter(x_mm,y_mm,c=values,marker='s',s=1,cmap='magma',vmin=0,vmax=1,rasterized=True)
                xm = np.mean(x_mm)
                ym = np.mean(y_mm)
                if mark_grainnumber:
                    plt.annotate(f'({i})',(xm,ym),fontsize=marksize)

                if num_print:
                    color = 'k' if value >= num_cut else 'r'
                    plt.annotate(f'({i}:{value:.2f})',(xm,ym),c=color,fontsize=marksize)

            plt.colorbar(sc,ax=ax)
            plt.xticks(x_ticks,fontsize=ticksize)
            plt.yticks(y_ticks,fontsize=ticksize)
            plt.xlabel(x_label,fontsize=labelsize)
            plt.ylabel(y_label,fontsize=labelsize)
            plt.axis('equal')

            if also_save:
                plt.savefig(self.directory+save_name,transparent=True,bbox_inches='tight')

            plt.show()

        else:
            print("plot_type must be 'orientation', 'comp', or 'acc'")







def rotate_matrix(mat, angles, axes=(1, 0), reshape=False):
    return np.stack([rotate(mat, angle, axes=axes, reshape=reshape) for angle in angles])
def rotmat(mu,om):
    Mom = np.array([[np.cos(-om), -np.sin(-om),0],
                   [np.sin(-om),np.cos(-om),0],
                   [0,0,1]])
    Mmu = np.array([[1,0,0],
                    [0,np.cos(mu),-np.sin(mu)],
                    [0,np.sin(mu),np.cos(mu)]])
    M = np.matmul(Mom,Mmu)
    return M                  

def pix2lab(cent_pix_y,cent_pix_z,pix_y,pix_z,pix_size,SDD,k0):
    #distance from beam center to pixle of interest in m
    y = SDD
    z = (pix_z-cent_pix_z) * pix_size
    x = (pix_y-cent_pix_y) * pix_size #y will be the sample detector distance
    #calculate the components of Q
    qx = k0*(x/np.sqrt(x**2+y**2+z**2))
    qy = k0*(1-y/np.sqrt(x**2+y**2+z**2))
    qz = k0*z/np.sqrt(x**2+y**2+z**2) 
    sign = math.copysign(1,qx) 
    HK = sign*math.sqrt(qx**2 + qy**2) 
    return qy,qx,qz,HK

#@numba.njit(parallel=False)
def compute_vmesh(h, k, l, deltaK, I_list,spots = [],bragg=False,):
    if bragg:
        Q=np.zeros((len(spots[0]),3))
        for i in range(len(spots[0])):
            o = spots[0][i]
            y = spots[1][i]
            x = spots[2][i]
            index_h = np.searchsorted(h, deltaK[y, x, o, 0], side='left')
            index_k = np.searchsorted(k, deltaK[y, x, o, 1], side='left')
            index_l = np.searchsorted(l, deltaK[y, x, o, 2], side='left')
            index_h = min(index_h, len(h) - 1)
            index_k = min(index_k, len(k) - 1)
            index_l = min(index_l, len(l) - 1)
            Q[i,:] = index_h,index_k,index_l
        return Q
    else:    
        vmesh = np.zeros((len(h), len(k), len(l)))
        #dmesh = np.zeros((len(h), len(k), len(l)))

        for i in range(len(I_list[0][0])):
            im = I_list[1][i]
            #im2 = I_list[2][i]
            o = I_list[0][0][i]
            x = I_list[0][1][i]
            y = I_list[0][2][i]
            index_h = np.searchsorted(h, deltaK[y, x, o, 0], side='left')
            index_k = np.searchsorted(k, deltaK[y, x, o, 1], side='left')
            index_l = np.searchsorted(l, deltaK[y, x, o, 2], side='left')

            index_h = min(index_h, len(h) - 1)
            index_k = min(index_k, len(k) - 1)
            index_l = min(index_l, len(l) - 1)

            if im>vmesh[index_h, index_k, index_l]:
                vmesh[index_h, index_k, index_l] = im

    return vmesh




from collections import deque

def is_valid_neighbor(x, y, z, shape):
    return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]

    def find_connected_component(arr, start, tolerance=0):
    DIRECTIONS = [(dx, dy, dz) for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if (dx, dy, dz) != (0, 0, 0)]
    stack = [(start, 0)]  
    visited = set()
    connected_component = []

    while stack:
        (x, y, z), zero_count = stack.pop()

        if (x, y, z) in visited:
            continue
        visited.add((x, y, z))

        connected_component.append((x, y, z))

        for dx, dy, dz in DIRECTIONS:
            nx, ny, nz = x + dx, y + dy, z + dz

            if is_valid_neighbor(nx, ny, nz, arr.shape):
                if arr[nx, ny, nz] != 0 and (nx, ny, nz) not in visited:
                    stack.append(((nx, ny, nz), 0))
                elif arr[nx, ny, nz] == 0 and zero_count < tolerance:
                    stack.append(((nx, ny, nz), zero_count + 1))  

    return connected_component


def expand_group(arr, group, tolerance=0):
    """Expand the existing group by searching for adjacent nonzero pixels that fit within tolerance."""
    DIRECTIONS = [(dx, dy, dz) for dx in [-1, 0, 1] 
                                   for dy in [-1, 0, 1] 
                                   for dz in [-1, 0, 1] 
                                   if (dx, dy, dz) != (0, 0, 0)]

    queue = deque(group)  # Start with the existing group
    expanded_group = set(group)  # Use a set for faster lookups

    while queue:
        x, y, z = queue.popleft()

        for dx, dy, dz in DIRECTIONS:
            nx, ny, nz = x + dx, y + dy, z + dz

            if is_valid_neighbor(nx, ny, nz, arr.shape) and (nx, ny, nz) not in expanded_group:
                if arr[nx, ny, nz] != 0:
                    queue.append((nx, ny, nz))
                    expanded_group.add((nx, ny, nz))
                elif arr[nx, ny, nz] == 0 and tolerance > 0:
                    queue.append((nx, ny, nz))
                    expanded_group.add((nx, ny, nz))
                    tolerance -= 1  # Allow a limited number of zeros

    return list(expanded_group)



