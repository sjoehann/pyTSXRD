# -*- coding: utf-8 -*-
"""
Created on Thu Nov 7 15:34:00 2024

@author: sjö

Class to load images for surface analysis 
"""

import sys, os, subprocess, pdb, re
import numpy as np
from numpy import float32
from datetime import datetime
import tifffile, fabio
import pickle, yaml, copy
import scipy, polarTransform
from skimage.transform import warp_polar
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges
from ImageD11 import columnfile, blobcorrector, peaksearcher
import pyTSXRD
from pyTSXRD.angles_and_ranges import merge_overlaps
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label, find_objects,generate_binary_structure

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"
class ImProcessor:
    def __init__(self,analysis=None,directory=None,name=None):
        self.log = []
        self.analysis = analysis
        self.background = None
        self.directory = directory if directory is not None else analysis.directory
        self.name = name
        self.sweeps = []
        self.processing = {'options':None,'image_rows':None,'eta_ranges':None,'thresholds':None}

    def add_to_log(self,str_to_add,also_print=False):
        self.log.append(str(datetime.now()) + '> ' + str_to_add)
        if also_print:
            print(str_to_add)

    def set_attr(self,attr,value):
        try:
            old = getattr(self,attr)
        except:
            old = None
        setattr(self,attr,value)
        new = getattr(self,attr)
        if attr == 'geometry' and old is not None:
            old = old.__dict__
        if attr == 'geometry' and new is not None:
            new = new.__dict__
        self.add_to_log(attr + ': ' + str(old) + ' -> ' + str(new))

    def add_to_attr(self,attr,value):
        try:
            old_list = getattr(self,attr)
        except:
            old_list = None
        if type(old_list) == list:
            setattr(self,attr,old_list + [value])
            new_list = getattr(self,attr)
            self.add_to_log(attr + ': += ' + str(new_list[-1]))
        else:
            raise AttributeError('This attribute is not a list!')

    def add_sweep(self,dy,omega_start=None,omega_step=None,directory=None,stem=None,ndigits='auto',ext=None,frames=None):
        sweep = {'dy':dy,'omega_start':omega_start,'omega_step':omega_step,'directory':directory,'stem':stem,'ndigits':ndigits,'ext':ext,'frames':frames}
        self.sweeps.append(sweep)

    def prepare_sweeps(self):
        prepared = []

        for sweep in self.sweeps:
            s = sweep.copy()
            all_files = sorted(os.listdir(s['directory']))
            matching_files = [f for f in all_files if f.startswith(s['stem']) and f.endswith(s['ext'])]

            if len(matching_files) == 0:
                raise FileNotFoundError(f'No files found for {s["directory"]}{s["stem"]}*{s["ext"]}')

            regex = re.compile(r'\d+')

            if s['ndigits'] == 'auto':
                stem_name = os.path.splitext(matching_files[0])[0]
                dig_part = regex.findall(stem_name)[-1]
                s['ndigits'] = len(dig_part)

            if s['frames'] is None:
                s['frames'] = list(range(len(matching_files)))

            if max(s['frames']) >= len(matching_files):
                raise FileNotFoundError(f'Some frames are missing in {s["directory"]}')

            n0 = int(matching_files[0].replace(s['stem'],'').replace(s['ext'],''))

            for i in range(max(s['frames']) + 1):
                expected_file = s['stem'] + str(n0 + i).zfill(s['ndigits']) + s['ext']
                if expected_file not in matching_files:
                    raise FileNotFoundError(expected_file)

            s['filenames'] = [matching_files[i] for i in s['frames']]
            s['omegas'] = np.array([s['omega_start'] + s['omega_step'] * i for i in s['frames']])

            prepared.append(s)

        self.set_attr('sweeps',prepared)
        self.add_to_log(f'Prepared {len(prepared)} sweeps.',True)

    def _get_image_path(self,dy_index,omega_index):
        target_omega = self.analysis.omega_range[omega_index]
        sweep = self.sweeps[dy_index]
        image_index = np.argmin(np.abs(sweep['omegas'] - target_omega))
        tolerance = abs(sweep['omega_step']) / 2 + 1e-6

        if abs(sweep['omegas'][image_index] - target_omega) > tolerance:
            raise ValueError(f'No image found at omega {target_omega} for dy {sweep["dy"]}')

        return os.path.join(sweep['directory'],sweep['filenames'][image_index])

    def calculate_background(self,max_images=50):
        backgrounds = []

        for dy_index in range(len(self.sweeps)):
            print(f'Calculating backgrounds: {round(100 * dy_index / max(len(self.analysis.y_range) - 1,1),1)}% done    ',end="\r")

            n_images = min(max_images,len(self.analysis.omega_range) - 2)
            omega_indices = np.unique(np.linspace(1,len(self.analysis.omega_range) - 2,n_images,dtype=int))
            config_file = self.analysis.directory + f"sweepparam_background_{dy_index}.yml"
            file_lines = ['image-files:\n  directory: ' + self.sweeps[dy_index]['directory'],'  files: "']	
            for omega_index in omega_indices:
                file_lines.append(self._get_image_path(dy_index,omega_index))

            file_lines.append('"\noptions:\n  empty frames: 0\n  max-frames: 0\nmeta:\n  omega:')

            with open(config_file,'w') as f:
                f.write('\n'.join(file_lines))

            imgs = imageseries.open(config_file,'image-files')
            backgrounds.append(calculate_bckg(imgs,list(range(len(imgs)))))

        backgrounds = np.array(backgrounds)
        self.set_attr("background",backgrounds)
        print('Background done!                                      ')
        return backgrounds

    def _get_thresholds(self,threshold):
        thresholds = np.array([threshold * np.mean(background) for background in self.background])
        self.processing['thresholds'] = thresholds
        return thresholds

    def _crop_image(self,img,image_rows=None):
        if image_rows is None:
            return img

        start = int(img.shape[0] * image_rows[0])
        stop = int(img.shape[0] * image_rows[1])
        return img[start:stop,:]


    def _make_eta_mask(self,shape,eta_ranges):
        if eta_ranges is None:
            return np.ones(shape,dtype=bool)

        yy,xx = np.indices(shape)
        cy = self.analysis.geometry.y_center
        cx = self.analysis.geometry.z_center

        dy = yy-cy
        dx = xx-cx

        eta = np.degrees(np.arctan2(dx,-dy))
        eta = (eta+180)%360-180

        mask = np.zeros(shape,dtype=bool)

        for eta_min,eta_max in eta_ranges:
            mask |= (eta >= eta_min) & (eta <= eta_max)

        return mask


    def _prepare_mask(self,eta_ranges,binning,image_rows):
        px = self.analysis.geometry.detz_size
        py = self.analysis.geometry.dety_size

        raw_mask = self._make_eta_mask((py,px),eta_ranges)

        if binning > 1:
            bpy = py//binning
            bpx = px//binning
            raw_mask = raw_mask[:bpy*binning,:bpx*binning]
            raw_mask = raw_mask.reshape(bpy,binning,bpx,binning).all(axis=(1,3))

        #mask = np.rot90(raw_mask,k=-1)
        mask = self._crop_image(raw_mask,image_rows)

        return mask

    def load_images(self,omega_index,thresholds,binning=1,image_rows=None,mask=None):
        px,py = self.analysis.geometry.detz_size,self.analysis.geometry.dety_size
        bpx = int(px // binning)
        bpy = int(py // binning)
        shape = (bpy,binning,bpx,binning)
        processed_imgs = []

        for dy_index in range(len(self.sweeps)):
            image_path = self._get_image_path(dy_index,omega_index)
            img = fabio.open(image_path).data.astype(np.float32) - self.background[dy_index]
            img = np.rot90(img.reshape(shape).mean(axis=(-1,1)),k=-1)
            img = self._crop_image(img,image_rows)

            if mask is not None:
                img[~mask] = 0

            img[img < thresholds[dy_index]] = 0
            processed_imgs.append(img.astype(np.float16))

        return np.array(processed_imgs)

    def _smooth3(self, img):
        p = np.pad(img, 1, mode="edge")
        return (
            p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
            p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
            p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
        ) / 9.0


    def _peak_markers(self, img, comp, base_th, min_peak_dist=2, min_support=4):
        dist = distance_transform_edt(comp)
        markers = np.zeros_like(comp, dtype=np.int32)
        centers = []

        p = np.pad(dist, 1, mode="edge")
        is_peak = (
            (dist >= p[:-2, :-2]) &
            (dist >= p[:-2, 1:-1]) &
            (dist >= p[:-2, 2:]) &
            (dist >= p[1:-1, :-2]) &
            (dist >= p[1:-1, 2:]) &
            (dist >= p[2:, :-2]) &
            (dist >= p[2:, 1:-1]) &
            (dist >= p[2:, 2:]) &
            comp &
            (dist > 1)
        )

        ys, xs = np.where(is_peak)

        if len(xs) == 0:
            ys_all, xs_all = np.where(comp)
            if len(xs_all) == 0:
                return markers, centers

            idx = np.argmax(dist[ys_all, xs_all])
            y = ys_all[idx]
            x = xs_all[idx]
            markers[y, x] = 1
            centers.append((x, y, 1))
            return markers, centers

        vals = dist[ys, xs]
        order = np.argsort(vals)[::-1]

        for idx in order:
            y = ys[idx]
            x = xs[idx]

            y0 = max(0, y - 1)
            y1 = min(comp.shape[0], y + 2)
            x0 = max(0, x - 1)
            x1 = min(comp.shape[1], x + 2)

            support = np.count_nonzero(comp[y0:y1, x0:x1])

            if support < min_support:
                continue

            too_close = False
            for cx, cy, lab in centers:
                dx = x - cx
                dy = y - cy
                if dx * dx + dy * dy < min_peak_dist * min_peak_dist:
                    too_close = True
                    break

            if too_close:
                continue

            lab = len(centers) + 1
            centers.append((x, y, lab))
            markers[y, x] = lab

        return markers, centers


    def _grow_from_markers(self, img, comp, markers):
        labels = markers.copy()

        ys, xs = np.where(comp)

        if len(xs) == 0:
            return labels

        dist = distance_transform_edt(comp)
        vals = dist[ys, xs]
        order = np.argsort(vals)[::-1]

        changed = True
        while changed:
            changed = False

            for idx in order:
                y = ys[idx]
                x = xs[idx]

                if labels[y, x] != 0:
                    continue

                y0 = max(0, y - 1)
                y1 = min(img.shape[0], y + 2)
                x0 = max(0, x - 1)
                x1 = min(img.shape[1], x + 2)

                nb = labels[y0:y1, x0:x1]
                labs = nb[nb > 0]

                if len(labs) == 0:
                    continue

                counts = np.bincount(labs)
                labels[y, x] = np.argmax(counts)
                changed = True

        remaining = comp & (labels == 0)

        ys_r, xs_r = np.where(remaining)

        if len(xs_r) == 0:
            return labels

        vals_r = img[ys_r, xs_r]
        order = np.argsort(vals_r)[::-1]

        changed = True
        while changed:
            changed = False

            for idx in order:
                y = ys_r[idx]
                x = xs_r[idx]

                if labels[y, x] != 0:
                    continue

                y0 = max(0, y - 1)
                y1 = min(img.shape[0], y + 2)
                x0 = max(0, x - 1)
                x1 = min(img.shape[1], x + 2)

                nb = labels[y0:y1, x0:x1]
                labs = nb[nb > 0]

                if len(labs) == 0:
                    continue

                counts = np.bincount(labs)
                labels[y, x] = np.argmax(counts)
                changed = True

        return labels


    def _split_component(self, img, comp_mask, base_th, size_th, min_peak_dist=4, min_support=4):
        comp_labels, ncomp = label(comp_mask)
        regions = []

        for comp_lab in range(1, ncomp + 1):
            comp = comp_labels == comp_lab
            ys_all, xs_all = np.where(comp)

            if len(xs_all) < size_th:
                continue

            markers, centers = self._peak_markers(
                img,
                comp,
                base_th,
                min_peak_dist=min_peak_dist,
                min_support=min_support)

            if len(centers) <= 1:
                intensities = img[ys_all, xs_all]
                peak_idx = np.argmax(intensities)

                regions.append({
                    "xs": xs_all,
                    "ys": ys_all,
                    "pixels": set(zip(xs_all, ys_all)),
                    "cx": xs_all[peak_idx],
                    "cy": ys_all[peak_idx],
                    "intensities": intensities,
                    "used": False,})

                continue

            assigned = self._grow_from_markers(img, comp, markers)

            for lab in range(1, len(centers) + 1):
                ys, xs = np.where(assigned == lab)

                if len(xs) < size_th:
                    continue

                intensities = img[ys, xs]
                peak_idx = np.argmax(intensities)

                regions.append({
                    "xs": xs,
                    "ys": ys,
                    "pixels": set(zip(xs, ys)),
                    "cx": xs[peak_idx],
                    "cy": ys[peak_idx],
                    "intensities": intensities,
                    "used": False,})

        return regions


    def select_signals(self,Images,offset,base_th,size_th,min_overlap=[0.8,0.8,0.6],min_peak_dist=4,min_support=4):
        signal_data = self.analysis.full_image
        signal_index = np.max(signal_data[0]) + 1 if len(signal_data[0]) > 0 else 0
        signal_start = signal_index

        omega_groups = []

        for om in range(Images.shape[0]):
            regions = []
            for dy in range(Images.shape[1]):
                img = Images[om,dy]
                dy_th = base_th[dy]

                if np.count_nonzero(img) == 0:
                    continue

                positive_vals = img[img > 0]

                if len(positive_vals) < size_th:
                    continue

                comp_mask = img >= dy_th
                regs = self._split_component(img,comp_mask,dy_th,size_th,min_peak_dist=min_peak_dist,min_support=min_support)

                for reg in regs:
                    reg["dy"] = dy
                    regions.append(reg)

            if not regions:
                continue

            reg_groups = [[regions[0]]]

            for reg in regions[1:]:
                found = False
                dy1 = reg["dy"]

                for gn, g in enumerate(reg_groups):
                    for gg in g:
                        dy2 = gg["dy"]

                        if dy1 == dy2 or abs(dy1 - dy2) > 2:
                            continue

                        dxc = reg["cx"] - gg["cx"]
                        dyc = reg["cy"] - gg["cy"]
                        dist = (dxc * dxc + dyc * dyc) ** 0.5

                        overlap = len(reg["pixels"] & gg["pixels"])
                        frac_reg = overlap / len(reg["pixels"])
                        frac_gg = overlap / len(gg["pixels"])

                        if (frac_reg >= min_overlap[0] or frac_gg >= min_overlap[0]) and dist <= 8:
                            reg_groups[gn].append(reg)
                            found = True
                            break

                    if found:
                        break

                if not found:
                    reg_groups.append([reg])

            for reg_g in reg_groups:
                pixels_by_dy = {}
                all_dys = set()

                for reg in reg_g:
                    dy = reg["dy"]
                    all_dys.add(dy)

                    if dy not in pixels_by_dy:
                        pixels_by_dy[dy] = set()

                    pixels_by_dy[dy].update(reg["pixels"])

                omega_groups.append({
                    "omega": om,
                    "regions": reg_g,
                    "pixels_by_dy": pixels_by_dy,
                    "dys": all_dys,
                })

        if not omega_groups:
            print("Number of assigned features: 0")
            self.analysis.set_attr("full_image", signal_data)
            return

        merged_groups = [[omega_groups[0]]]

        for grp in omega_groups[1:]:
            found = False
            om1 = grp["omega"]

            for mg in merged_groups:
                for gg in mg:
                    om2 = gg["omega"]

                    if om1 == om2 or abs(om1 - om2) > 1:
                        continue

                    shared_dys = grp["dys"] & gg["dys"]

                    if not shared_dys:
                        continue

                    dy_frac_1 = len(shared_dys) / len(grp["dys"])
                    dy_frac_2 = len(shared_dys) / len(gg["dys"])

                    if dy_frac_1 < min_overlap[1] and dy_frac_2 < min_overlap[1]:
                        continue

                    overlap_sum = 0
                    total_pix_1 = 0
                    total_pix_2 = 0

                    for dy in shared_dys:
                        p1 = grp["pixels_by_dy"][dy]
                        p2 = gg["pixels_by_dy"][dy]
                        overlap_sum += len(p1 & p2)
                        total_pix_1 += len(p1)
                        total_pix_2 += len(p2)

                    if total_pix_1 == 0 or total_pix_2 == 0:
                        continue

                    pix_frac_1 = overlap_sum / total_pix_1
                    pix_frac_2 = overlap_sum / total_pix_2

                    if pix_frac_1 >= min_overlap[2] or pix_frac_2 >= min_overlap[2]:
                        mg.append(grp)
                        found = True
                        break

                if found:
                    break

            if not found:
                merged_groups.append([grp])

        for final_group in merged_groups:
            all_x = []
            all_y = []

            for omega_group in final_group:
                for reg in omega_group["regions"]:
                    all_x.extend(reg["xs"])
                    all_y.extend(reg["ys"])

            all_x = np.array(all_x)
            all_y = np.array(all_y)

            if len(all_x) == 0:
                continue

            for omega_group in final_group:
                for reg in omega_group["regions"]:
                    for x, y, intensity in zip(reg["xs"], reg["ys"], reg["intensities"]):
                        signal_data[0].append(signal_index)
                        signal_data[1].append(float(intensity))
                        signal_data[2].append(int(x))
                        signal_data[3].append(int(y))
                        signal_data[4].append(int(reg["dy"]))
                        signal_data[5].append(int(omega_group["omega"]) + offset)

            signal_index += 1

        print(f'Number of assigned features: {signal_index - signal_start}')
        self.analysis.set_attr("full_image", signal_data)


    def process_images(self,block=10,threshold=0.5,size_th=10,binning=1,image_rows=None,
                       eta_ranges=None,min_overlap=[0.8,0.8,0.6],min_peak_dist=4,min_support=4):
        if self.background is None:
            self.calculate_background()

        thresholds = self._get_thresholds(threshold)
        mask = self._prepare_mask(eta_ranges,binning,image_rows)
        num_images = len(self.analysis.omega_range)
        self.analysis.set_attr('full_image',[[],[],[],[],[],[]])

        for block_number,start_ind in enumerate(range(1,num_images - 1,block)):
            end_ind = min(start_ind + block,num_images - 1)
            print(single_separator + f'\nLoading block {block_number}\n' + single_separator)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.load_images,j,thresholds,
                                           binning,image_rows,mask) for j in range(start_ind,end_ind)]
                Images = []

                for image_index,future in enumerate(futures):
                    try:
                        Images.append(future.result())
                        print(f'Preparing images for TSXRD: {round(100 * (image_index + 1) / len(futures))}% done    ',end="\r")
                    except Exception as e:
                        logging.error(f"Error processing omega index {start_ind + image_index}: {e}")
                        raise

            Images = np.array(Images,dtype=np.float16)
            print("Images done!                                                   ")
            self.select_signals(Images,start_ind,thresholds,size_th,min_overlap,
                                min_peak_dist=min_peak_dist,min_support=min_support)
            del Images

        self.analysis.set_attr("full_image",np.array(self.analysis.full_image))
        print(f"Total number of features: {len(np.unique(self.analysis.full_image[0]))}")



def calculate_bckg(imgs, indices):
    """Calculates background using median image for a subset of images."""
    try: sub_set = np.asarray( [imgs[i] for i in indices] )
    except: raise ValueError(' - incorrect indices!')
    I = [img.mean() for img in imgs]
    norm  = np.max(I)/np.mean(I) # This corrects for non-uniformities in the sweep's total intensity profile
    return norm*np.median(sub_set,axis=0).astype(np.float32)