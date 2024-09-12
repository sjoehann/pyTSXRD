# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:00:00 2022

@author: sjö

Class to perform for full analysis from raw images to grain map
"""

import sys, os, subprocess, copy, pickle
import tifffile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyTSXRD
from pyTSXRD.p212_tools import load_p212_log, load_p212_fio, parse_p212_command
from pyTSXRD.angles_and_ranges import convert_ranges, mod_360, disorientation
from pyTSXRD.GvectorEvaluator import merge_GE_list
import scipy
from scipy.ndimage import binary_erosion,binary_dilation
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
#import set_GrainSpotter,set_PolySim

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"


class DataAnalysis:

    def __init__(self, directory=None, name=None):
        self.log = []
        self.directory = None
        self.name = None
        self.material = None
        self.pressure = None
        self.temperature = None
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.sweepProcessors = []
        self.peakIndexers = []
        self.yml_det_order = []
        self.gvectorEvaluator = None
        self.grainSpotter = None
        self.grains = []
#         self.parents = []
        self.pos_tol = None
        self.ang_tol = None
        self.sample_pix_x = np.linspace(-3.5, 3.5, 701) #sample size and pixels used when creating map
        self.sample_pix_y = np.linspace(-3.5, 3.5, 701)
        self.beamsize = 0.1 #meamsize in mm
        self.plot_range = []
        self.label_range = []
        self.add_to_log('Initialized DataAnalysis object.', False)
        if directory:
            self.set_attr('directory', directory)
        if name:
            self.set_attr('name', name)
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
        print(double_separator + 'DataAnalysis object:')
        print('directory:', self.directory)
        print('name:', self.name)
#         print('parents:', len(self.parents))
        print('material:', self.material)
        print('pressure:', self.pressure)
        print('temperature:', self.temperature)
        print('position:', self.position)
        print('rotation:', self.rotation)
        print('sweepProcessors:', len(self.sweepProcessors))
        print('peakIndexers:', len(self.peakIndexers))
        print('yml_det_order:', self.yml_det_order)
        print('gvectorEvaluator:', self.gvectorEvaluator)
        print('grainSpotter:', self.grainSpotter)
        print('grains:', len(self.grains))
        print('pos_tol:', self.pos_tol)
        print('ang_tol:', self.ang_tol)
        print('sample_pix_x', [self.sample_pix_x[0],self.sample_pix_x[-1],len(self.sample_pix_x)])
        print('sample_pix_y', [self.sample_pix_y[0],self.sample_pix_y[-1],len(self.sample_pix_y)])
        print('beamsize', self.beamsize)
        print('plot_range', self.plot_range)
        print('label_range', self.label_range)
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log:
                print(record)
        return

    def save_geometries_as_yml(self, yml_det_order=None):
        if yml_det_order:
            self.set_attr('yml_det_order', yml_det_order)
        gs = [self.sweepProcessors[ind - 1].geometry for ind in self.yml_det_order]
        yml_file = pyTSXRD.Geometry.save_geometries_as_yml(
            gs, self.directory, self.name + '.yml', overwrite=True)
        self.add_to_log('Exported all the geometries as .yml file: ' +yml_file,False)
        return

    def process_images(self,frames=None,save_tifs=False,q0_pos=None,rad_ranges=None,thr=None,plot_proj=False):
        """Method for loading images and removing background"""
        for SP in self.sweepProcessors:
            #print(single_separator + f'\nPROCESSING for {SP.name}:')
            if not os.path.exists(SP.directory):
                # Create the directory if does not exist.
                os.makedirs(SP.directory)

            # Loading and processing images:
            if type(frames) == list:
                SP.chunk['frames'] = frames  # If substack is needed.
            # Loading and processing images usually takes SOME time.
            SP.load_sweep()
            if save_tifs:
                SP.save_tifs()  # Usually not needed.
            # bckg = 'auto50' - median over equally spaced images (not more
            # than ~50).
            SP.process_imgs(bckg='auto50')

            # Usually takes LONG time.
            SP.calculate_projections(q0_pos=q0_pos, rad_ranges=rad_ranges)
            if plot_proj:
                SP.plot()
            if not thr or thr == 'auto':
                SP.calculate_peaksearch_thrs()
                thr = min(SP.peaksearch_thrs)
            SP.export_data(thr=thr)

            imgs = SP.imgs
            SP.set_attr('imgs', None)  # As images are too heavy to save
            pickle.dump(SP,open(SP.directory + SP.name + "_SweepProcessor.p","wb"))
            SP.set_attr('imgs', imgs)

    def peaksearch(self,peaksearch_thrs, peakmerge_thrs,min_peak_dist,use_temp_tifs=False, del_temp_tifs=False):
        """Runing ImageD11 peaksearching function"""
        for SP in self.sweepProcessors:
            #print(single_separator + f'\nPEAKSEARCHING for {SP.name}:')
            yaml_file = SP.save_peaksearch_yaml(thresholds=peaksearch_thrs,pix_tol=min_peak_dist,spline_path=SP.geometry.spline_file)
            SP.run_peaksearcher(yaml_file=yaml_file,use_imgs=False,use_temp_tifs=use_temp_tifs,del_temp_tifs=del_temp_tifs)  # Usually takes LONG time.
            SP.run_peakmerger(yaml_file = yaml_file, thresholds = peakmerge_thrs)
            # As images are too heavy and not needed futhermore.
            SP.set_attr('imgs', None)
            SP.dump_arrays()
            pickle.dump(SP,open(SP.directory + SP.name +"_SweepProcessor.p","wb"))

    def index(self, move_det_xyz_mm=[0, 0, 0], thr = 0):
        if not move_det_xyz_mm:
            move_det_xyz_mm = [0, 0, 0]
        self.set_attr('peakIndexers', [])
        GE_list = []
        
        for iP, SP in enumerate(
                self.sweepProcessors):    # (e.g. Varex 1,2,3,4).
            #print(single_separator + f'\nINDEXING of {SP.name}:')
            # Evaluating peaks:
            PI = SP.generate_PeakIndexer()
            if move_det_xyz_mm != [0, 0, 0]:
                PI.geometry.move_detector(move_det_xyz_mm)
#             PI.set_attr('name', PI.name + '_' + PI.geometry.material['name'])
            PI.run_indexer() # This time indexer is run for computing tth values
            try:
                peaks = [p for p in PI.peaks if p['IMax_int'] > thr(p['tth'])]
                PI.add_to_log(f'Removing {len(PI.peaks)-len(peaks)} peaks that are below tth-dependent threshold. {len(peaks)} left.', False)
                PI.set_attr('peaks', peaks )
            except:
                peaks = [p for p in PI.peaks if p['IMax_int'] > thr]
                PI.add_to_log(f'Removing {len(PI.peaks)-len(peaks)} peaks that are below tth-independent threshold. {len(peaks)} left.', False)
                PI.set_attr('peaks', peaks )
            PI.set_attr('name', PI.name+'_fltrd')
#             print(PI.log[-2], '\n',  PI.log[-1])
            PI.run_indexer()  # This time indexer is run on filtered peaks
            GE_list.append(PI.generate_GvectorEvaluator())

            pickle.dump(PI,open(PI.directory + PI.name + "_PeakIndexer.p", "wb"))
            self.add_to_attr('peakIndexers', PI)
            
        GE = merge_GE_list(self.directory, self.name, GE_list, spot3d_id_reg = 100*1000)
        self.set_attr('gvectorEvaluator', GE)
        self.save_geometries_as_yml()
        
    def reindex(self, move_det_xyz_mm=[0, 0, 0]):
        if not move_det_xyz_mm:
            move_det_xyz_mm = [0, 0, 0]
        GE_list = []
        peakIndexers = copy.copy(self.peakIndexers)
        self.set_attr('peakIndexers', [])
        
        for iP, PI in enumerate(peakIndexers):    # (e.g. Varex 1,2,3,4).
            #print(single_separator + f'\nRE-INDEXING of {PI.name}:')
            PI.geometry.set_attr('distance', PI.sweepProcessor.geometry.distance) # Reset detector position (undo previous movings)
            PI.geometry.set_attr('y_center', PI.sweepProcessor.geometry.y_center) # Reset detector position (undo previous movings)
            PI.geometry.set_attr('z_center', PI.sweepProcessor.geometry.z_center) # Reset detector position (undo previous movings)
                            
            if move_det_xyz_mm != [0, 0, 0]:
                PI.geometry.move_detector(move_det_xyz_mm)
            PI.run_indexer()
            GE_list.append(PI.generate_GvectorEvaluator())
            self.add_to_attr('peakIndexers', PI)

            pickle.dump(PI,open(PI.directory + PI.name +"_PeakIndexer.p", "wb"))
            
        GE = merge_GE_list(self.directory, self.name, GE_list, spot3d_id_reg = 100*1000)
        self.set_attr('gvectorEvaluator', GE)
        
        
    def evaluateGvectors(self,ds_tol=None,tth_gap=1,ds_gap=0.1,eta_gap=1,omega_gap=None,to_plot=False,save_arrays=False):
        #print(single_separator + f'\nEVALUATING GVECTORS for {self.name}:')
        if ds_tol == 'auto':
            ds_tol = None
        self.gvectorEvaluator.remove_not_inrings(ds_tol=ds_tol)
        if not omega_gap:
            try:
                omega_gap = 2 * abs(self.gvectorEvaluator.merged[0].peakIndexer.sweepProcessor.sweep['omega_step'])
            except:
                omega_gap = 2 * abs(self.gvectorEvaluator.merged[0].merged[0].peakIndexer.sweepProcessor.sweep['omega_step'])
        self.gvectorEvaluator.calculate_ranges(tth_gap=tth_gap,ds_gap=ds_gap,eta_gap=eta_gap,omega_gap=omega_gap)
#             GE.group_gvectors(0.1, 1, 1)
        self.gvectorEvaluator.calc_histo(0.5, 0.5, plot=to_plot, save_arrays=save_arrays)
        self.gvectorEvaluator.save_gve(overwrite=True)
        pickle.dump(self.gvectorEvaluator,open(self.directory +self.name +"_GvectorEvaluator.p","wb"))
        self.gvectorEvaluator.set_attr('ds_eta_omega', np.zeros((1, 1, 1)))  # To save memory

    def searchGrains(self, grainSpotter=None):
        """Method to run grainspotter"""
        #print(single_separator + f'\nRUNNING GRAINSPOTTER for {self.name}')
        if grainSpotter:
            self.set_attr('grainSpotter', grainSpotter)
        self.grainSpotter.set_attr('spacegroup', self.material['spacegroup'])
        self.grainSpotter.set_attr('directory', self.directory)
        self.grainSpotter.set_attr('ini_file', self.name + '.ini')
        
        try:
            omega_step = abs(self.gvectorEvaluator.merged[0].peakIndexer.sweepProcessor.sweep['omega_step'])
        except:
            omega_step = abs(self.gvectorEvaluator.merged[0].merged[0].peakIndexer.sweepProcessor.sweep['omega_step'])
        self.grainSpotter.set_attr('domega', omega_step)

        # Usually takes SOME time.
        self.grainSpotter.run_grainspotter(gve_file=self.name + '.gve',log_file=self.name + '.log')
        # Below we load .log file (more parameters but lower precision)
        # and .gff (fewer parameters but higher precision) and merge them
        grains_log = pyTSXRD.Grain.load_log(self.directory, self.name+'.log')
        grains_gff = pyTSXRD.Grain.load_gff(self.directory, self.name+'.gff')
        for gl in grains_log:
            gf = [g for g in grains_gff if g.grain_id == gl.grain_id][0]
            gl.set_attr('mean_IA', gf.mean_IA)
            gl.set_attr('pos_chisq', gf.pos_chisq)
            gl.set_attr('position', gf.position)
            gl.set_attr('u', gf.u)
            gl.set_attr('ubi', gf.ubi)
        self.set_attr('grains', grains_log)
        
        pickle.dump(self.grainSpotter,open(self.directory +self.name +"_GrainSpotter.p","wb"))
        for g in self.grains:
            g.identify_measured(self.gvectorEvaluator.gvectors)

    def runPolyXSim(self, polyxsim, GE_SIM_list,also_plot=False):
        #print(single_separator + f'\nRUNNING POLYXSIM for {self.name}')
        GS = self.grainSpotter
        PS = polyxsim
        sim_dir = self.directory + 'sim_' + self.material['name'] + '/'
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)  # Create the directory if does not exist
        subprocess.call('rm -r ' + sim_dir, shell=True)
        PS.set_attr('directory', sim_dir)

        for g in self.grains:
            #print(single_separator + f'\nGRAIN {g.grain_id}')
            if not g.size:
                g.set_attr('size', 0.05)
#             g.identify_measured(self.gvectorEvaluator.gvectors)
            GE_list = []
            for GE in GE_SIM_list:
                PI = GE.peakIndexer
                det_num = int(PI.name.split('_d')[-1].split('_')[0])
                PS.set_attr('inp_file', f'g{g.grain_id:03d}_d{det_num}.inp')
                PS.set_attr('stem', f'g{g.grain_id:03d}_d{det_num}')
                PS.set_attr('geometry', copy.copy(PI.geometry))
                PS.set_attr('grains', [g])
                PS.save_inp(overwrite=True)
                d = min(0.2,np.sqrt(g.position[0]**2 +g.position[1]**2 +g.position[2]**2))
                PS.set_attr('sample_cyl', [2 * d, 2 * d])
                PS.run_PolyXsim()  # Usually takes SOME time.
                GE_this_det = pyTSXRD.GvectorEvaluator(directory=PS.direc)
                GE_this_det.set_attr('peakIndexer', copy.copy(pyTSXRD.PeakIndexer()))
                GE_this_det.set_attr('material', dict(PS.geometry.material))
                GE_this_det.peakIndexer.set_attr('geometry', copy.copy(PS.geometry))
                GE_this_det.load_gve(gve_file=PS.stem + '.gve')
                for gv in GE_this_det.gvectors:
                    gv['stage_x'] = PI.sweepProcessor.position[0]
                    gv['stage_y'] = PI.sweepProcessor.position[1]
                    gv['stage_z'] = PI.sweepProcessor.position[2]
                GE_list.append(GE_this_det)
            GE_sim = merge_GE_list(sim_dir, f'{g.grain_id:03d}', GE_list, spot3d_id_reg = 100*1000)
            GE_sim.calc_histo(0.5, 0.5, plot=False, save_arrays=False)
            GE_sim.remove_not_ranges(GS.ds_ranges,GS.tth_ranges,GS.omega_ranges,convert_ranges(GS.eta_ranges))
            # Sort in asceding ds
            ind = np.argsort([g['ds'] for g in GE_sim.gvectors])
            GE_sim.save_gve(gve_file=f'g{g.grain_id:03d}.gve', overwrite=True)
            g.set_attr('expected_gvectors', [GE_sim.gvectors[i] for i in ind])
            if also_plot:
                g.plot_measured_vs_expected(directory = self.directory+'grains/')  # Usually takes LONG time.

    def mark_peaks(self):
        #print(single_separator + f'\nMARKING GRAIN PEAKS for {self.name}')
        gvectors = []
        for G in self.grains:
            gvectors += G.measured_gvectors

        ind = np.argsort([g['omega'] for g in gvectors])  # Sort in asceding omega
        gvectors = [gvectors[i] for i in ind]

        peaks = []

        for g in gvectors:
            spot3d_id_reg = self.gvectorEvaluator.spot3d_id_reg
            GE_id = int(g['spot3d_id'] / self.gvectorEvaluator.spot3d_id_reg)
            spot3d_id = g['spot3d_id'] - \
                self.gvectorEvaluator.spot3d_id_reg * GE_id
            if GE_id == 0:
                GE = self.gvectorEvaluator
            else:
                GE = self.gvectorEvaluator.merged[GE_id - 1]
            PI = pickle.load(open(GE.directory +GE.name +'_PeakIndexer.p',"rb"))
            PI_id = int(spot3d_id / PI.spot3d_id_reg)
            if PI_id > 0:
                PI = PI.absorbed[PI_id - 1]
                spot3d_id = spot3d_id - int(spot3d_id / PI.spot3d_id_reg)
            p_list = [p for p in PI.peaks if p['spot3d_id'] == spot3d_id]

            if len(p_list) == 0:
                raise ValueError('peak for gvector ' +
                                 str(g['spot3d_id']) + ' not found!')
            elif len(p_list) == 1:
                peaks.append(p_list[0])
            else:
                raise ValueError('peak for gvector ' +
                                 str(g['spot3d_id']) + ' more than 1 record found!')

        #print(f'Peaks to mark: {len(peaks)}')
        for SP in self.sweepProcessors:
            SP_FULL = pickle.load(open(SP.directory +SP.name +"_SweepProcessor.p","rb"))
            omegas = [(omg[0] + omg[1]) / 2 for omg in SP_FULL.chunk['omegas']]
            omgeta_rngs = SP_FULL.projs['omgeta_rngs']
            immax_rngs = SP_FULL.projs['immax_rngs']
            ranges = SP_FULL.projs['ranges']

            for p in peaks:
                d_omg = abs(np.asarray(omegas) - p['omega'])
                omg_ind = np.where(d_omg == min(d_omg))[0][0]
#                 eta_ind = round(g['eta']-180)
                r = [p['fc'] - SP.geometry.y_center, p['sc'] - SP.geometry.z_center]
                thh_ind = round(np.sqrt(r[0]**2 + r[1]**2))
                eta_ind = round(mod_360(np.arctan2(r[1],r[0]) * 180 / np.pi, 180))
                this_rng = [rng for rng in ranges if rng[0]< thh_ind and rng[1] > thh_ind][0]
                rng_ind = ranges.index(this_rng)
                omgeta_rngs[rng_ind][omg_ind, eta_ind] = -1
                immax_rngs[round(p['fc']), round(p['sc'])] = -1

            path = SP.directory + 'projs_ranges/' + SP.name
            SP_FULL.add_to_log('Writing file: ' + path + "_immax_rngs_marked.tif", False)
            SP.add_to_log('Writing file: ' + path + "_immax_rngs_marked.tif",False)
            tifffile.imsave(path + "_immax_rngs_marked.tif", immax_rngs)
            for ir, rng in enumerate(ranges):
                SP_FULL.add_to_log('Writing file: ' + path + "_omgeta_rngs_marked.tif", False)
                SP.add_to_log('Writing file: ' + path + "_omgeta_rngs_marked.tif",False)
                tifffile.imsave(path + f"_omgeta_rngs_marked-{rng[0]}-{rng[1]}.tif",omgeta_rngs[ir])
            pickle.dump(
                SP_FULL,
                open(SP.directory + SP.name + "_SweepProcessor.p","wb"))

        return

    
    def remove_duplicates(self, ang_tol, pos_tol):
        """Finds grains that are close to each other in positions and orientations
         In such pairs duplicate is the one with worse metrics. This function removes such grains"""
        #print(double_separator+f'\nRemoving duplicates in {len(self.grains)} grains')
        #print('Tolerances [ang, pos]: ', [ang_tol, pos_tol])
        n_grains = len(self.grains)
        if not ang_tol   >= 0: raise ValueError('ang_tol must be non-negative!')
        if not pos_tol   >= 0: raise ValueError('pos_tol must be non-negative!')
        self.set_attr('ang_tol', ang_tol)
        self.set_attr('pos_tol', pos_tol)
        checked_indices = set()
        list_of_grains = []
        for i1, g1 in enumerate(self.grains):
            if i1 not in checked_indices:
                duplicates = [g1]
                checked_indices.add(i1)
                for i2, g2 in enumerate(self.grains):
                    if i2 > i1 and i2 not in checked_indices:
                        d_ang = disorientation(g1.u, g2.u, sgno=self.material['spacegroup'])
                        #print(d_ang,ang_tol)
                        if d_ang <= ang_tol:
                            d_pos = np.linalg.norm( np.asarray(g1.position) - np.asarray(g2.position) )
                            #print(d_pos,pos_tol)
                            if d_pos <= pos_tol:
                                duplicates.append(g2)
                                checked_indices.add(i2)
                if len(duplicates) > 1:
                    quality = [len(g.measured_gvectors) / g.mean_IA for g in duplicates]
                    best_index = np.argmax(quality)
                    best_grain = duplicates[best_index]
                    list_of_grains.append(best_grain)
                else:
                    list_of_grains.append(g1)
            else:
                print('Removing grain:', i1)
        self.set_attr('grains', list_of_grains)
        print('Number of grains:',len(self.grains))
        print('Removed',n_grains-len(self.grains),'grains')   
    
    
    def plot_dmap(self,sample_pix_x,sample_pix_y,grain_number=0,plot_type = 'full',also_save=False,with_colorbar=False):
        """plotting for testing maps and making figures"""
        if grain_number <= len(self.grains):
            d_map = self.grains[grain_number].d_map
            d_map_smoothed = scipy.ndimage.gaussian_filter(d_map, 10/2.355)
            support = d_map_smoothed > 0.5*np.amax(d_map_smoothed)
            support_smoothed = scipy.ndimage.gaussian_filter(np.float32(support), 10/2.355)
            f = d_map_smoothed*(support_smoothed > 0.5)
            np.amax(d_map_smoothed)
            d_map_thresholded = d_map_smoothed > 0.5*np.amax(d_map_smoothed)
            if plot_type == 'full':
                fig = plt.figure(figsize=(16, 10))
                sub1 = fig.add_subplot(221)
                plt.title('density = number of gvectors')
                plt.imshow(np.rot90(d_map), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                if with_colorbar:
                    plt.colorbar()
                sub2 = fig.add_subplot(222)
                plt.imshow(np.rot90(d_map_smoothed), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                plt.title('smoothed')
                if with_colorbar:
                    plt.colorbar()
                sub3 = fig.add_subplot(223)
                plt.imshow(np.rot90(support), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                plt.title('support')
                if with_colorbar:
                    plt.colorbar()
                sub4 = fig.add_subplot(224)
                plt.imshow(np.rot90(f), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                plt.title('result')
                if with_colorbar:
                    plt.colorbar()
                plt.show()
            elif plot_type == 'simple':
                fig = plt.figure(figsize=(10, 6),dpi=500)
                sub1 = fig.add_subplot(121)
                plt.title('Density (number of G-vectors)')
                plt.imshow(np.rot90(d_map), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                plt.xlabel('x (mm)',fontsize=15)
                plt.ylabel('y (mm)',fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                if with_colorbar:
                    plt.colorbar()
                sub2 = fig.add_subplot(122)
                plt.imshow(np.rot90(f), extent = [sample_pix_x[0], sample_pix_x[-1], sample_pix_y[0], sample_pix_y[-1]])
                plt.title('Grain')
                plt.xlabel('x (mm)',fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.ylabel('')
                if also_save:
                    plt.savefig(self.directory+f'densitymap_{grain_number}.png',transparent=True,bbox_inches='tight')
                plt.show()
            else: 
                print('Plot type is not available. Pick full or simple')
        else:
            raise ValueError('Grain number not in range!') 
            
    def make_grainmatrix(self,completeness_th=0.5,also_plot=False,save_matrix=False):
        """Make all completenes maps into matrix+remove very weak grains"""
        n_grains = len(self.grains)
        list_of_grains = []
        for i,g in enumerate(self.grains):
            g.make_array(completeness_th=completeness_th,also_plot=also_plot)
            if np.sum(g.matrix)>0:
                list_of_grains.append(g)
        self.set_attr('grains', list_of_grains)
        print('Number of grains:',len(self.grains))
        print('Removed',n_grains-len(self.grains),'grains') 
        if save_matrix:
            maps=np.zeros((len(self.grains),np.shape(g.matrix)[0],np.shape(g.matrix)[1]),dtype=float)
            for i,g in enumerate(self.grains):
                maps[i,:,:] = g.matrix
            N1, N2, N3 = maps.shape
            output_file = open(self.directory+f'maps_{N3}x{N2}x{N1}.raw', 'wb')
            np.float32(maps).tofile(output_file)
            output_file.close()
        
    def apply_samplemask(self,radius,also_plot=False,plot_invers=False,save_invers=False):
        """Remove parts of grains outside of the sample boundries"""
        for i,g in enumerate(self.grains):
            g.sample_mask(radius)
            if also_plot:
                plt.figure(figsize=(4,4))
                plt.imshow(np.rot90(maps[i]))
                plt.show()
            if plot_invers:
                g.plot_colors_singlegrains(self.plot_range,self.label_range,also_save=save_invers)
        
    def map_sample(self,also_plot=False,grain_th=10):
        """remove pixels where there is overlap, only keep for the most intence"""
        n_grains=len(self.grains)
        list_of_grains = []
        bad_grains = []
        # Create a mask array where elements are True if they are equal to the maximum value
        maps = np.zeros((len(self.grains),np.shape(self.grains[0].matrix)[0],np.shape(self.grains[0].matrix)[1]),dtype=float)
        for i,g in enumerate(self.grains):
            maps[i,:,:] = g.matrix#/g.mean_IA
        max_values = np.max(maps, axis=0)
        mask = maps == max_values
        maps = maps*mask                
        for i,g in enumerate(self.grains):
            g.set_attr('matrix',maps[i])
            g.set_attr('index',np.nonzero(maps[i]))
            if also_plot:
                plt.figure(figsize=(4,4))
                plt.imshow(np.rot90(maps[i]))
                plt.show()
            if np.shape(g.index)[1] > grain_th:
                list_of_grains.append(g)
            else:
                bad_grains.append(g)
        for g in bad_grains: #instead of having a 0 pixel in the middle of the sample we assign it to the sorounding grain.
            #these are just grains that are below what we can measure so it makes sence to remove these false grains
            if len(g.index[0])>0:
                non_zero_mask = g.matrix != 0
                structuring_element = np.ones((5, 5), dtype=bool)  # 5x5 structuring element for dilation
                edge_mask = binary_dilation(non_zero_mask, structure=structuring_element)
                edge_mask = np.invert(non_zero_mask)*edge_mask
                new_index = np.argmax(np.sum(edge_mask*maps, axis=(1,2)))
                map_new = g.matrix+self.grains[new_index].matrix
                self.grains[new_index].set_attr('matrix',map_new)
                self.grains[new_index].set_attr('index',np.nonzero(map_new))
        self.set_attr('grains', list_of_grains)
        print('Number of grains:',len(self.grains))
        print('Removed',n_grains-len(self.grains),'grains')   
    
    def tilt_correction(self,ang_inc=0):
        """Corrects the grain orientationfor the angle of incidence """
        ang_rad = np.radians(ang_inc)
        #rotational matrix 
        rotmat = np.array([[1,0,0],
                           [0,np.cos(ang_rad),-np.sin(ang_rad)],
                           [0,np.sin(ang_rad),np.cos(ang_rad)]])
        for g in self.grains:
            #print(g.phi)
            u_start = g.u
            u_new = np.dot(rotmat,u_start)
            ori = Orientation.from_matrix(u_new)
            ipfkey = plot.IPFColorKeyTSL(symmetry.Oh)
            ori.symmetry = ipfkey.symmetry
            eul = ori.to_euler(degrees=True)
            #print(eul[0])
            g.set_attr('phi',[-eul[0][2],eul[0][1],eul[0][0]])

    def plot_colors(self,save_name,plot_type='full',also_save=False,mark_grainnumber=False,mark_millers=False,show_pos=False):
        """Plot surface as inverse pole figure map"""
        if plot_type == 'full':
            euler_list = [g.phi for g in self.grains]
            #fig=plt.figure(dpi=500)
            plt.rcParams["axes.grid"] = False
            ori = Orientation.from_euler(np.deg2rad(euler_list))
            ipfkey = plot.IPFColorKeyTSL(symmetry.Oh)
            ori.symmetry = ipfkey.symmetry
            color_matrix = ipfkey.orientation2color(ori)
            ori.scatter("ipf", c=color_matrix, direction=ipfkey.direction)
            plt.title('')
            if also_save:
                plt.savefig(self.directory+f'colortriangle_{save_name}.png', facecolor='white', bbox_inches='tight', transparent=True)
                
            figure, ax = plt.subplots(figsize=(10,10),dpi=250)
            ax.set_xticks(self.plot_range)
            ax.set_xticklabels(self.label_range,fontsize=25)
            ax.set_yticks(self.plot_range)
            ax.set_yticklabels(self.label_range,fontsize=25)
            ax.set_xlabel('x (mm)',fontsize=25)
            ax.set_ylabel('y (mm)',fontsize=25)
            for i,g in enumerate(self.grains):
                plt.scatter(g.index[0],g.index[1],color=color_matrix[i],marker='s', s=1)
                if mark_grainnumber:
                    plt.annotate(f'({i})', (int(np.mean(g.index[0])), int(np.mean(g.index[1]))),fontsize=12)
                if mark_millers:
                    plt.annotate(f'({int(g.miller[0])},{int(g.miller[1])},{int(g.miller[2])})',
                                 (int(np.mean(g.index[0])),int(np.mean(g.index[1]))),fontsize=12)
                if show_pos:
                    plt.scatter(np.mean(g.index[0]),np.mean(g.index[1]),c='k')
            if also_save:
                plt.savefig(self.directory+f'colormap_{save_name}.png',transparent=False,bbox_inches='tight')
            plt.show()
        elif plot_type == 'single':
            for g in self.grains:
                g.plot_colors_singlegrains(self.plot_range,self.label_range,also_save=False)
        else:
            print('Plot type is not available. Pick full or single')
            
            
    def find_closest_points(self, grainnum1,grainnum2, threshold=10):
        matrix1 = np.column_stack(self.grains[grainnum1].index)
        matrix2 = np.column_stack(self.grains[grainnum2].index)

        distances = np.linalg.norm(matrix1[:, np.newaxis, :] - matrix2, axis=2)
        if np.any(distances < threshold):
            return 1
        else:
            return 0
                
    def combine_duplicates(self, ang_tol, pos_tol):
        """combine grains that are one grain split into parts"""
        n_grains = len(self.grains)
        if ang_tol < 0:
            raise ValueError('ang_tol must be non-negative!')
        if pos_tol < 0:
            raise ValueError('pos_tol must be non-negative!')

        self.set_attr('ang_tol', ang_tol)
        self.set_attr('pos_tol', pos_tol)

        checked_indices = set()
        list_of_grains = []
        
        for i1, g1 in enumerate(self.grains):
            if i1 not in checked_indices:
                duplicates = [g1]
                checked_indices.add(i1)
                for i2, g2 in enumerate(self.grains):
                    if i2 > i1 and i2 not in checked_indices:
                        d_ang = disorientation(g1.u, g2.u, sgno=self.material['spacegroup'])
                        #print(d_ang,ang_tol)
                        if d_ang <= ang_tol:
                            d_pos = self.find_closest_points(i1, i2, threshold=pos_tol)
                            #print(d_pos,pos_tol)
                            if d_pos > 0:
                                duplicates.append(g2)
                                checked_indices.add(i2)
                if len(duplicates) > 1:
                    for i3, g3 in enumerate(duplicates):
                            for i4, g4 in enumerate(self.grains):
                                if i4 not in checked_indices:
                                    d_ang = disorientation(g3.u, g4.u, sgno=self.material['spacegroup'])
                                    #print(d_ang,ang_tol)
                                    if d_ang <= ang_tol:
                                        d_pos = self.find_closest_points(i3, i4, threshold=pos_tol)
                                        #print(d_pos,pos_tol)
                                        if d_pos > 0:
                                            duplicates.append(g4)
                                            checked_indices.add(i4)
                    quality = [len(g.measured_gvectors) / g.mean_IA for g in duplicates]
                    best_index = np.argmax(quality)
                    best_grain = duplicates[best_index]
                    combined_matrix = sum(g.matrix for g in duplicates)
                    best_grain.set_attr('matrix', combined_matrix)
                    best_grain.set_attr('index', np.nonzero(combined_matrix))
                    list_of_grains.append(best_grain)
                else:
                    list_of_grains.append(g1)
        self.set_attr('grains', list_of_grains)
        print('Number of grains:',len(self.grains))
        print('Removed',n_grains-len(self.grains),'grains')   
        

def plot_sinogram(name, list_DATApaths):
    import tifffile
    omegas = []
    ypos_s = []
    for path in list_DATApaths:
        #print("PATH:", path)
        DATA = pickle.load(open(path, "rb"))
        SP = DATA.sweepProcessors[0]
        GE = DATA.gvectorEvaluator
        GS = DATA.grainSpotter
        GE.remove_not_ranges(GS.ds_ranges,GS.tth_ranges,GS.omega_ranges,GS.eta_ranges)
        # -1*0.5*SP.sweep['omega_step'] # useful for checking positions
        omg_sys_err = 0
        omegas = omegas + [g['omega'] + omg_sys_err for g in GE.gvectors]
        ypos_s = ypos_s + [DATA.position[1] for g in GE.gvectors]

    fig = plt.figure(figsize=(10, 5))
    plt.scatter(omegas, ypos_s, s=1)

    omgy = np.zeros([len(SP.chunk['omegas']) + 2,
                    len(list_DATApaths) + 2], dtype=np.float32)
    omg_min = min(omegas)
    omg_stp = abs(SP.sweep['omega_step'])
    y_min = min(ypos_s)
    y_stp = (max(ypos_s) - min(ypos_s)) / (len(list_DATApaths) - 1)
    for omg, y in zip(omegas, ypos_s):
        omg_ind = (omg - omg_min) / omg_stp
        y_ind = (y - y_min) / y_stp
        omgy[round(omg_ind), round(y_ind)] += 1
    directory = DATA.directory.replace('/' + DATA.directory.split('/')[-2], '')
    name = DATA.name.replace('_'.join(DATA.name.split('_')[0:2]), name)
    tifffile.imsave(directory + name + '_sinogram.tif', omgy)

def compute_sinogram(name, list_DATApaths, pos_step = 'auto', omg_step = 'auto', eta_step = 'auto',
                     ds_ranges = 'auto', tth_ranges = 'auto',
                     omega_ranges = 'auto', eta_ranges = 'auto',save_sinogram=False):
    omg_list = []
    eta_list = []
    pos_list = []
    for path in list_DATApaths:
        #print("PATH:", path)
        DATA = pickle.load(open(path, "rb"))
        SP = DATA.sweepProcessors[0]
        GE = DATA.gvectorEvaluator
        GS = DATA.grainSpotter
        this_ds_ranges = GS.ds_ranges if ds_ranges == 'auto' else ds_ranges
        this_tth_ranges = GS.tth_ranges if tth_ranges == 'auto' else tth_ranges
        this_omega_ranges = GS.omega_ranges if omega_ranges == 'auto' else omega_ranges
        this_eta_ranges = GS.eta_ranges if eta_ranges == 'auto' else eta_ranges
        GE.remove_not_ranges(
            this_ds_ranges,
            this_tth_ranges,
            this_omega_ranges,
            this_eta_ranges)
        omg_offs = 0 # -1*0.5*SP.sweep['omega_step'] # useful for checking positions
        omg_list = omg_list + [g['omega'] + omg_offs for g in GE.gvectors]
        eta_list = eta_list + [g['eta']              for g in GE.gvectors]
        pos_list = pos_list + [g['stage_y']          for g in GE.gvectors]
        # pos_list = pos_list + [DATA.position[1]      for g in GE.gvectors]

    omg_min = min(omg_list)
    eta_min = min(eta_list)
    pos_min = min(pos_list)
    
    if type(omg_step) == float:
        omg_stp = abs(omg_step)
    else:
        omg_stp = abs(SP.sweep['omega_step'])
 
    if type(eta_step) == float:
        eta_stp = abs(eta_step)
    else:
        eta_stp = 1.0
    
    if type(pos_step) == float:
        pos_stp = abs(pos_step)
    else:
        if max(pos_list) > min(pos_list):
            pos_stp = (max(pos_list) - min(pos_list)) / (len(list_DATApaths) - 1)
            # pos_stp = (max(pos_list) - min(pos_list)) / (len( set(pos_list)) - 1)
        else:
            pos_stp = 0.01
    
    omg_pts = 1 + round( (max(omg_list) - min(omg_list)) / omg_stp )
    eta_pts = 1 + round( (max(eta_list) - min(eta_list)) / eta_stp )
    pos_pts = 1 + round( (max(pos_list) - min(pos_list)) / pos_stp )    
    
    omg_linspace = np.linspace(omg_min, omg_min+omg_stp*(omg_pts-1), omg_pts)
    eta_linspace = np.linspace(eta_min, eta_min+eta_stp*(eta_pts-1), eta_pts)
    pos_linspace = np.linspace(pos_min, pos_min+pos_stp*(pos_pts-1), pos_pts)

    fig = plt.figure(figsize=(10, 5))
    plt.xlabel('omega (deg)')
    plt.ylabel('y (mm)')
    plt.scatter(omg_list, pos_list, s=1)
    
    sinogram = np.zeros([pos_pts, omg_pts, eta_pts], dtype=np.float32)
    for ig in range(len(omg_list)):
        omg_ind = round( (omg_list[ig] - omg_min) / omg_stp )
        eta_ind = round( (eta_list[ig] - eta_min) / eta_stp )
        pos_ind = round( (pos_list[ig] - pos_min) / pos_stp )
        sinogram[pos_ind, omg_ind, eta_ind] += 1
    directory = DATA.directory.replace('/' + DATA.directory.split('/')[-2], '')
    name = DATA.name.replace('_'.join(DATA.name.split('_')[0:2]), name)
                    
    import tifffile
    tifffile.imsave(directory + name + '_sinogram.tif', np.sum(sinogram,axis=2))
    if save_sinogram:
        N1, N2, N3 = sinogram.shape
        print('Saving file:', f'_sinogram_eta-omg-y_{N3}x{N2}x{N1}.raw')
        output_file = open(DATA.directory+DATA.name+f'_sinogram_eta_omg-y_{N3}x{N2}x{N1}.raw', 'wb')
        np.float32(sinogram).tofile(output_file)
        output_file.close()
    return sinogram, pos_linspace, omg_linspace, eta_linspace


def set_MultiDATA(directory, name, list_DATApaths, num_0):
    def new_name(old_name):
        return old_name.replace('_'.join(old_name.split('_')[0:2]), name)

    path = list_DATApaths[num_0]
    #print("PATH:", path)
    DATA = pickle.load(open(path, "rb"))
    DATA.directory = directory
    DATA.name = new_name(DATA.name)

    for ii, SP in enumerate(DATA.sweepProcessors):
        DATA.sweepProcessors[ii].directory = directory
        DATA.sweepProcessors[ii].name = new_name(DATA.sweepProcessors[ii].name)
        DATA.sweepProcessors[ii].geometry.directory = directory
        DATA.sweepProcessors[ii].geometry.par_file = new_name(
            DATA.sweepProcessors[ii].geometry.par_file)

    for ii, PI in enumerate(DATA.peakIndexers):
        DATA.peakIndexers[ii].directory = directory
        DATA.peakIndexers[ii].name = new_name(DATA.peakIndexers[ii].name)
        DATA.peakIndexers[ii].geometry.directory = directory
        DATA.peakIndexers[ii].geometry.par_file = new_name(
            DATA.peakIndexers[ii].geometry.par_file)

    DATA.gvectorEvaluator.directory = directory
    DATA.gvectorEvaluator.name = new_name(DATA.gvectorEvaluator.name)
    for ii, GM in enumerate(DATA.gvectorEvaluator.geometries):
        DATA.gvectorEvaluator.geometries[ii].directory = directory
        DATA.gvectorEvaluator.geometries[ii].par_file = new_name(
            DATA.gvectorEvaluator.geometries[ii].par_file)

    max_ds_tol = 0
    max_detshift = 0
    min_distance = 999999999999999999
    for path in list_DATApaths:
        if path == list_DATApaths[num_0]:
            continue
        DATA_ = pickle.load(open(path, "rb"))
        detshift = list(np.asarray(DATA.position) - np.asarray(DATA_.position))
        DATA_.reindex(move_det_xyz_mm=detshift)
        DATA.gvectorEvaluator.absorb(DATA_.gvectorEvaluator)
        max_ds_tol = max(max_ds_tol, DATA_.gvectorEvaluator.ds_tol)
        max_detshift = max(max_detshift, np.linalg.norm(detshift))
        min_distance = min(min_distance, min(
            [gm.distance for gm in DATA_.gvectorEvaluator.geometries]))

    #print(max_ds_tol)
    # 1000 to convert detshift from mm to microns
    extra_tth = np.degrees(np.arctan(1000 * max_detshift / min_distance))
    extra_ds_tol = DATA.gvectorEvaluator.geometries[0].ds_from_tth(extra_tth)
    DATA.gvectorEvaluator.set_attr('ds_tol', max_ds_tol + extra_ds_tol)

    return DATA


def merge_DATA_list(base_DATA, list_DATApaths, spot3d_id_reg = None):
    baseDATA = copy.copy(base_DATA)
    baseDATA.set_attr('peakIndexers', [])
    baseDATA.set_attr('sweepProcessors', [])

    max_ds_tol = 0
    max_detshift = 0
    min_distance = 999999999999999999
    GE_list = []
    for path in list_DATApaths:
        thisDATA = pickle.load(open(path, "rb"))
        detshift = list(np.asarray(thisDATA.position) - np.asarray(baseDATA.position) )
        thisDATA.reindex(move_det_xyz_mm=detshift)
        GE_list.append(thisDATA.gvectorEvaluator)
        max_ds_tol = max(max_ds_tol, thisDATA.gvectorEvaluator.ds_tol)
        max_detshift = max(max_detshift, np.linalg.norm(detshift))
    baseDATA.set_attr('gvectorEvaluator', merge_GE_list(baseDATA.directory, baseDATA.name, GE_list, spot3d_id_reg) )
    min_distance = min(min_distance, min([GE.merged[0].peakIndexer.geometry.distance for GE in GE_list]))
#     print(max_ds_tol)
    extra_tth = np.degrees(np.arctan(1000 * max_detshift / min_distance)) # 1000 to convert detshift from mm to microns
    extra_ds_tol = baseDATA.gvectorEvaluator.merged[0].merged[0].peakIndexer.geometry.ds_from_tth(extra_tth)
    baseDATA.gvectorEvaluator.set_attr('ds_tol', max_ds_tol + extra_ds_tol)

    return baseDATA



