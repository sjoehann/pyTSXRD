# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 11:00:00 2022

@author: sjö

Class to work with a grain object. Loads .log  files.
"""

import sys, os, subprocess, pdb, re, copy
import numpy as np
import pyTSXRD
import scipy
from numpy import float32
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import gcd
from ImageD11.transform import compute_g_vectors
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from skimage.transform import radon, iradon

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"

class Grain:
    
    def __init__(self, directory=None, grain_id=None):
        self.log = []
#         self.absorbed =[]
        self.directory = None
        self.grain_id = None
        self.log_file = None
        self.gff_file = None
        self.spacegroup = None
        self.unitcell = []
        self.u = None
        self.ubi = None
        self.eps = None
        self.size = None
        self.mean_IA = None
        self.position = None
        self.pos_chisq = None
        self.r = None
        self.phi = None
        self.quaternion = None
        self.summary = None
        self.gvectors_report = []
        self.measured_gvectors = []
        self.expected_gvectors = []
        self.matrix = None
        self.index = None
        self.miller = [0,0,0]
        self.add_to_log('Initialized Grain object.', False)
        if directory: self.set_attr('directory', directory)
        if grain_id : self.set_attr('grain_id' , grain_id)
        return
 

    def add_to_log(self, str_to_add, also_print = False):
        self.log.append( str(datetime.now()) + '> ' + str_to_add )
        if also_print: print(str_to_add)
        return
    
    
    def set_attr(self, attr, value):
        """Method to set an attribute to the provided value and making a corresponding record in the log."""
        try:
            old = getattr(self, attr)
        except:
            old = None
        setattr(self, attr, value)
        new = getattr(self, attr)
        if attr in ['gvectors_report', 'measured_gvectors', 'expected_gvectors']:
            old, new = f'list of {len(old)}', f'list of {len(new)}'
        self.add_to_log(attr+': '+str(old)+' -> '+str(new))
        return
        
        
    def add_to_attr(self, attr, value):
        """Method to append value to the choosen attribute. The attribute must be a list."""
        try:
            old_list = getattr(self, attr)
        except:
            old_list = None
        setattr(self, attr, old_list+[value])
        new_list = getattr(self, attr)
        self.add_to_log(attr+': += '+str(new_list[-1]))
        return
    
    
    def print(self, also_log = False):
        print(double_separator+'Grain object:')
        print('directory:' , self.directory )
        print('grain_id:'  , self.grain_id  )
        print('log_file:'  , self.log_file  )
        print('gff_file:'  , self.gff_file  )
        print('spacegroup:', self.spacegroup)
        print('unitcell:'  , self.unitcell  )
        print('u:'         , self.u         )
        print('ubi:'       , self.ubi       )
        print('eps:'       , self.ubi       )
        print('size:'      , self.size      )
        print('mean_IA:'   , self.mean_IA   )
        print('position:'  , self.position  )
        print('pos_chisq:' , self.pos_chisq )
        print('r:'         , self.r         )
        print('phi:'       , self.phi       )
        print('quaternion:', self.quaternion)
        print('summary:'   , self.summary   )
        print('gvectors_report:'  , len(self.gvectors_report  ))
        print('measured_gvectors:', len(self.measured_gvectors))
        print('expected_gvectors:', len(self.expected_gvectors))
#         print('absorbed:', len(self.absorbed))
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return

    
    def load_log(directory, log_file):
        print(double_separator + 'Reading file: ' + directory + log_file)
        if not os.path.isfile(directory + log_file): raise FileNotFoundError

        header = []
        f = open(directory + log_file,"r")
        line = f.readline()
        while line and '#  gvector_id' not in line:
            if 'Found' in line and 'grains' in line: # the very 1st line in file
                n_grains = int(line.split()[1])
            elif 'Syntax' in line or line == '.':
                pass
            else:
                header.append(line)
            line = f.readline()
        if '#  gvector_id' in line:
            titles = line
        else:
            raise ValueError('Title line not found!') 
        list_of_grains = []
        line = f.readline()
        while line:
            if 'Grain ' in line:
                num_grain = line.split()[1].replace(',','')
                g = Grain(directory, int(num_grain))
                g.set_attr('log_file', log_file)

                line = f.readline()
                keys = header[0].replace('#', '').split()
                values = [int(x) for x in line.split()]
                g.set_attr('summary', dict(zip( keys, values )) )

                line = f.readline()
                values = [float(x) for x in line.split()]
                g.set_attr('mean_IA'  , values[0]  )
                g.set_attr('position' , values[1:4])
                g.set_attr('pos_chisq', values[4]  )

                line = f.readline()
                raw_0 = [float(x) for x in line.split()]
                line = f.readline()
                raw_1 = [float(x) for x in line.split()]
                line = f.readline()
                raw_2 = [float(x) for x in line.split()]
                g.set_attr('u', np.array([raw_0,raw_1,raw_2]) )

                line = f.readline()

                line = f.readline()
                raw_0 = [float(x) for x in line.split()]
                line = f.readline()
                raw_1 = [float(x) for x in line.split()]
                line = f.readline()
                raw_2 = [float(x) for x in line.split()]
                g.set_attr('ubi', np.array([raw_0,raw_1,raw_2]) )

                line = f.readline()
                line = f.readline()
                g.set_attr('r', [float(x) for x in line.split()])

                line = f.readline()
                line = f.readline()
                g.set_attr('phi', [float(x) for x in line.split()])

                line = f.readline()
                line = f.readline()
                g.set_attr('quaternion', [float(x) for x in line.split()])               

                line = f.readline()
                keys = titles.replace('#', '').split()
                gvectors_report = []
                while True:
                    line = f.readline()
                    values = [float(x) if '.' in x else int(x) for x in line.split()]
                    if len(keys) == len(values)-1:
                        gvectors_report.append( dict(zip(keys,values[1:])) )
                    else:
                        break
                g.set_attr('gvectors_report', gvectors_report)
                list_of_grains.append(g)
            else:
                line = f.readline()
        f.close()
        print(f'{len(list_of_grains)} grains loaded.\n')
        return list_of_grains
    

    def load_gff(directory, gff_file):
        #print(double_separator + 'Reading file: ' + directory + gff_file)
        if not os.path.isfile(directory + gff_file): raise FileNotFoundError

        list_of_grains = []
        titles = "grain_id mean_IA chisq x y z U11 U12 U13 U21 U22 U23 U31 U32 U33 UBI11 UBI12 UBI13 UBI21 UBI22 UBI23 UBI31 UBI32 UBI33".split()
        with open(directory+gff_file, "r") as f:
            for line in f:
                if line[0] == '#' and 'grain_id' in line:
                    titles = line[1:-1].split()
                elif len(line.split()) == len(titles):
                    x = dict(zip(titles,line.split()))
                    g = Grain(directory, int(x['grain_id']))
                    g.set_attr('gff_file' , gff_file)
                    g.set_attr('mean_IA'  , float(x['mean_IA']) )
                    g.set_attr('pos_chisq', float(x['chisq']) )
                    g.set_attr('position' , [float(x['x']), float(x['y']), float(x['z'])] )
                    raw_0 = [float(v) for v in [x['U11'], x['U12'], x['U13']] ]
                    raw_1 = [float(v) for v in [x['U21'], x['U22'], x['U23']] ]
                    raw_2 = [float(v) for v in [x['U31'], x['U32'], x['U33']] ]
                    g.set_attr('u', np.array([raw_0,raw_1,raw_2]) )
                    raw_0 = [float(v) for v in [x['UBI11'], x['UBI12'], x['UBI13']] ]
                    raw_1 = [float(v) for v in [x['UBI21'], x['UBI22'], x['UBI23']] ]
                    raw_2 = [float(v) for v in [x['UBI31'], x['UBI32'], x['UBI33']] ]
                    g.set_attr('ubi', np.array([raw_0,raw_1,raw_2]) )
                    list_of_grains.append(g)
        f.close()                             
        print(f'{len(list_of_grains)} grains loaded.\n')
        return list_of_grains
    
    
    def save_gff(directory, gff_file, list_of_grains = None, overwrite = False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.add_to_log('Created directory: '+directory, False)
        #print(f'Writing file: {directory+gff_file}', True)

        while os.path.isfile(directory+gff_file):
            #print('File already exist!')
            if overwrite:
                break
            else:
                x = input('Type new name or ! to overwrite, a - abort:')
                if x in ['!']:
                    #print('Overwriting...')
                    break
                elif x in ['a', 'A']:
                    #print('Aborted!')
                    return
                else:
                    gff_file = x
                    
        titles = "grain_id mean_IA chisq x y z U11 U12 U13 U21 U22 U23 U31 U32 U33 UBI11 UBI12 UBI13 UBI21 UBI22 UBI23 UBI31 UBI32 UBI33".split()
        f = open(directory+gff_file ,"w") 
        f.write('# '+' '.join(titles) + '\n')
        for g in list_of_grains:
            s1 = ["{:d}".format(g.grain_id)]
            s2 = ["{:0.7f}".format(v) for v in [g.mean_IA, g.pos_chisq]+g.position]
            s3 = ["{:0.12f}".format(v) for v in g.u.flatten().tolist()]
            s4 = ["{:0.12f}".format(v) for v in g.ubi.flatten().tolist()]
            f.write(' '.join(s1+s2+s3+s4) + '\n' )
        f.close()
        #print('File closed!')
        return
    
    
    def identify_measured(self, gvectors):
        measured_gvectors = []
        for gv in self.gvectors_report:
            gv_list = list(filter(lambda g: g['spot3d_id'] == gv['peak_id'], gvectors))
            if len(gv_list) < 0:
                #print(gv)
                raise ValueError('this g-vector not found in the provided list of gvectors!')
            elif len(gv_list) > 1:
                #print(gv)
                raise ValueError('more than 1 g-vector was found in the provided list of gvectors!')
            else:
                measured_gvectors.append(gv_list[0])
        self.set_attr('measured_gvectors', measured_gvectors)
        return 
        
    def simulate_gvectors(self, geometry, omega_range, tth_range, beamflux, bckg, psf, peakshape):
        P = pyTSXRD.PolySim(directory = self.directory)
        P.set_attr('geometry', geometry)
        P.set_attr('beamflux', beamflux)
        P.set_attr('direc', './sim/')
        P.set_attr('stem', 'sim')
        P.set_attr('grains', [self])
         
        P.set_attr('omega_start', omega_range.start)
        P.set_attr('omega_step', omega_range.step)
        P.set_attr('omega_end', omega_range.end)
        P.set_attr('theta_min', tth_range.start)
        P.set_attr('theta_max', tth_range.end)
        P.set_attr('no_grains', 1)
        P.set_attr('gen_U', 0)
        P.set_attr('gen_pos', [0, 0])
        P.set_attr('gen_eps', [0, 0, 0 ,0, 0])
        P.set_attr('gen_size', [0.0, 0.0, 0.0 ,0.0])
        P.set_attr('sample_cyl', [0.17, 0.2])
        if not P.grains[0].size:
            P.grains[0].set_attr('size',0.05)
        P.set_attr('make_image', 0)
        P.set_attr('output', ['.tif', '.par', '.gve'])
        P.set_attr('bg', bckg)
        P.set_attr('psf', psf)
        P.set_attr('peakshape', peakshape) #[1, 4, 0.5])
        P.save_inp(inp_file = 'sim.inp')
        return
        
        
    def plot_measured_vs_expected(self, directory = None):
        if not directory: directory = self.directory
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.add_to_log('Created directory: '+directory, False)
        tth_measured = [gv['tth']   for gv in self.measured_gvectors]
        eta_measured = [gv['eta']   for gv in self.measured_gvectors]
        omg_measured = [gv['omega'] for gv in self.measured_gvectors]
        tth_expected = [gv['tth']   for gv in self.expected_gvectors]
        eta_expected = [gv['eta']   for gv in self.expected_gvectors]
        omg_expected = [gv['omega'] for gv in self.expected_gvectors]
        
        fig = plt.figure(figsize=(8, 6))
        sub1 = fig.add_subplot(131)
        sub1.scatter(omg_expected, eta_expected, s=10, c='k', marker="o", label='Expected')
        sub1.scatter(omg_measured, eta_measured, s=5 , c='r', marker="v", label='Measured')
        # sub1.legend(bbox_to_anchor=(0.4, 1.1))
        plt.title(f'Expected {len(self.expected_gvectors)} (black), measured {len(self.measured_gvectors)} (red)')
        plt.xlabel('omega (deg)')
        plt.ylabel('eta (deg)')

        sub2 = fig.add_subplot(132)
        sub2.scatter(tth_expected, omg_expected, s=10, c='k', marker="o", label='Expected')
        sub2.scatter(tth_measured, omg_measured, s=5 , c='r', marker="v", label='Measured')
        # sub1.legend(bbox_to_anchor=(0.4, 1.1))
        plt.title(f'Expected {len(self.expected_gvectors)} (black), measured {len(self.measured_gvectors)} (red)')
        plt.xlabel('tth (deg)')
        plt.ylabel('omega (deg)')

        sub3 = fig.add_subplot(133)
        sub3.scatter(tth_expected, eta_expected, s=10, c='k', marker="o", label='Expected')
        sub3.scatter(tth_measured, eta_measured, s=5 , c='r', marker="v", label='Measured')
        # sub1.legend(bbox_to_anchor=(0.4, 1.1))
        plt.title(f'Expected {len(self.expected_gvectors)} (black), measured {len(self.measured_gvectors)} (red)')
        plt.xlabel('tth (deg)')
        plt.ylabel('eta (deg)')
        
        plt.show()
        fig.savefig(directory+self.log_file.replace('.log', '')+'_g'+str(self.grain_id)+'_scatter.png')
        self.add_to_log('Saved measured vs expected: '+self.log_file.replace('.log', '')+'_g'+str(self.grain_id)+'_scatter.png', False)
        return

    
    def compute_density_map(self, GE_list, x, y, beamsize, omg_tol, eta_tol, tth_tol, support_thr,sample_rot=0,final_map=False,bigbeam=False):
        """Computed density and support maps on x,y grid. G-vectors are searched from a list of GvectorEvaluators.
        They are filtered according to tolerances and then also filters out gvectors that do not overlap with the support."""
        beamsize_in_pix = beamsize / np.sqrt((x[1]-x[0])**2 + (y[1]-y[0])**2)

        gvm_list = [] # list of measured g-vectors within (omg, tth, eta) tolerances
        exp_ind = []
        for GE in GE_list:
            wvl = GE.peakIndexer.geometry.wavelength
            wdg = GE.peakIndexer.geometry.wedge
            chi = GE.peakIndexer.geometry.chi
            for gvm in GE.gvectors:
                for i,gve in enumerate(self.expected_gvectors):
                    if abs(gvm['omega']-gve['omega']) > omg_tol: continue
                    if abs(gvm[ 'eta' ]-gve[ 'eta' ]) > eta_tol: continue
                    if abs(gvm[ 'tth' ]-gve[ 'tth' ]) > tth_tol: continue

                    GE_spotted, p_spotted = GE.trace_peak_by_spot3d_id(gvm['spot3d_id'])
                    g_spotted = copy.copy(gvm)
                    # g_spotted['spot3d_id'] = p_spotted['spot3d_id']
                    g_spotted['avg_intensity'] = p_spotted['avg_intensity']
                    wvl = GE_spotted.peakIndexer.geometry.wavelength
                    wdg = GE_spotted.peakIndexer.geometry.wedge
                    chi = GE_spotted.peakIndexer.geometry.chi
                    gve_g = compute_g_vectors([gve['tth']], [gve['eta']], [gve['omega']], wvl, wdg, chi)[0]
                    gvm_g = np.asarray([ gvm['gx'], gvm['gx'], gvm['gx'] ])
                    g_spotted['dg'] = np.linalg.norm(gve_g - gvm_g)
                    gvm_list.append(g_spotted)
                    exp_ind.append(i)

        # weights = []
        d_map = np.zeros([len(x), len(y)], float)
        d_map_g = np.zeros([len(gvm_list),len(x), len(y)], float)
        for i,gvm in enumerate(gvm_list):
            # weights.append( np.sqrt( np.sqrt(gvm['avg_intensity'])*gvm['tth']/ (0.25+gvm['dg']) ) )
            ind = abs(y - gvm['stage_y']) < beamsize/2
            d_map_g[i,:,ind] += 1
            d_map_g[i] = rotate(d_map_g[i], -(gvm['omega']+sample_rot), axes=(1, 0), reshape=False)
            d_map += d_map_g[i]
        support = gaussian_filter(d_map, beamsize_in_pix/2.355) > support_thr
        if bigbeam:
            gvectors = [gvm for gvm in gvm_list]
        else:
            gvectors =[]
            for i,gvm in enumerate(gvm_list):
                overlap = d_map_g[i]*(support > 0)
                if overlap.sum() > beamsize_in_pix**2:
                    gvectors.append(gvm)
        if final_map:
            d_map_c = np.zeros((len(self.expected_gvectors),len(x),len(y)))
            for i,gvm in enumerate(gvm_list):
                d_map_c[exp_ind[i]] += d_map_g[i]
            d_map_c[d_map_c>2.0] = 2.0
            d_map_c[d_map_c<1.0] = 0.0
            #print(np.round(np.sum(om_c)/len(self.expected_gvectors),2),np.round(len(self.measured_gvectors)/len(self.expected_gvectors),2))
            d_map = np.sum(d_map_c,axis=0)
            #print(om_c)
        self.set_attr('d_map', d_map)
        self.set_attr('support', support)
        self.set_attr('gvm_list', gvm_list)
        self.set_attr('gvectors', gvectors)
        return
    
    def make_array(self,completeness_th=0.5,also_plot=False):
        """Method of extracting the grain making a completeness map-array"""
        len_exp = len(self.expected_gvectors )
        d_map_smoothed = scipy.ndimage.gaussian_filter(self.d_map, 10/2.355)
        if np.max(self.d_map)/len_exp <= completeness_th-0.15: #remove grains with intensity that is very low 
            grainmap = 0*d_map_smoothed
        else:
            C = d_map_smoothed/len_exp
            support = C > (completeness_th*np.max(C)) #mask of C>th*C_max
            support_smoothed = scipy.ndimage.gaussian_filter(np.float32(support), 10/2.355) 
            r = np.sqrt(len(np.nonzero(d_map_smoothed*support_smoothed)[0])/np.pi) #approximate radius of the grain
            grainmap = (C * (support_smoothed > completeness_th))/np.sqrt((np.max(C)*np.sqrt(r)))
        if also_plot:
            plt.figure(figsize=(4,4))
            plt.imshow(np.rot90(grainmap))
            plt.colorbar()
            plt.show()
        self.set_attr('matrix', grainmap)
        self.set_attr('index', np.nonzero(grainmap))
        self.euler2miller()
        return
    
    def sample_mask(self,radius):
        """removes values outside of edges of a round sample. Note that the radious is in pixels and not mm."""
        shape = np.shape(self.matrix)
        rows, cols = np.ogrid[:shape[0], :shape[1]]
        center_row, center_col = shape[0] // 2, shape[1] // 2
        distances = np.sqrt((rows - center_row)**2 + (cols - center_col)**2)
        # Create a mask where values within the radius are True, and False otherwise
        mask = distances <= radius
        grainmap = self.matrix*mask
        self.set_attr('matrix', grainmap)
        self.set_attr('index', np.nonzero(grainmap))
        
    def plot_colors_singlegrains(self,plot_range,label_range,also_save=False):
        """Will plot the inverse pole figure map of one grain"""
        plt.rcParams["axes.grid"] = False
        ori = Orientation.from_euler(np.deg2rad(self.phi))
        ipfkey = plot.IPFColorKeyTSL(symmetry.Oh)

        ori.symmetry = ipfkey.symmetry
        color_matrix = ipfkey.orientation2color(ori)
        ori.scatter("ipf", c=color_matrix, direction=ipfkey.direction)
        plt.title('')
        if also_save:
            plt.savefig(self.directory+f'colortriangle_{self.grain_id}.png',transparent=False,bbox_inches='tight')


        figure, ax = plt.subplots(figsize=(6,6))
        plt.xlim(0,np.shape(self.matrix)[0])
        plt.ylim(0,np.shape(self.matrix)[1])
        ax.set_xticks(plot_range)
        ax.set_xticklabels(label_range,fontsize=15)
        ax.set_yticks(plot_range)
        ax.set_yticklabels(label_range,fontsize=15)
        plt.scatter(self.index[0],self.index[1],color=color_matrix,s=1)
        ax.set_xlabel('x (mm)',fontsize=15)
        ax.set_ylabel('y (mm)',fontsize=15)
        if also_save:
            plt.savefig(self.directory+f'colordmap_{self.grain_id}.png',transparent=True,bbox_inches='tight')
        plt.show()
    
    def euler2miller(self,maxindex=4): 
        """Function for calculating approximate miller index for a grain"""
        phi1=np.deg2rad(self.phi[0])
        PHI=np.deg2rad(self.phi[1])
        phi2=np.deg2rad(self.phi[2])

        g=np.array([[ np.cos(phi1)*np.cos(phi2)-np.sin(phi1)*np.sin(phi2)*np.cos(PHI),
                     np.sin(phi1)*np.cos(phi2)+np.cos(phi1)*np.sin(phi2)*np.cos(PHI), np.sin(phi2)*np.sin(PHI)],
                    [-np.cos(phi1)*np.sin(phi2)-np.sin(phi1)*np.cos(phi2)*np.cos(PHI),
                     np.sin(phi1)*np.sin(phi2)+np.cos(phi1)*np.cos(phi2)*np.cos(PHI), np.cos(phi2)*np.sin(PHI)],
                    [ np.sin(phi1)*np.sin(PHI), -np.cos(phi1)*np.sin(PHI), np.cos(PHI) ]])
       
        hkl1=np.sort(np.round(np.abs(g.T[2])/np.max(np.abs(g.T[2]))*maxindex))
        gcdnumber=gcd(int(hkl1[0]), gcd(int(hkl1[1]), int(hkl1[2])))
        millers = (hkl1/gcdnumber)[::-1].astype(int)
        self.set_attr('miller', millers)
        #print(millers)
        
    
    
        
    
    