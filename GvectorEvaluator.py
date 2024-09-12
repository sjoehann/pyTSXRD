 # -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:00:49 2022

@author: sjö

Class to work with a list of gvectors and *.gve files.
"""
import os, subprocess, copy
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from datetime import datetime
import pyTSXRD
from pyTSXRD.angles_and_ranges import mod_360
from pyTSXRD.angles_and_ranges import merge_overlaps, group_to_chains, convert_ranges
from ImageD11 import indexing # might require running "module load maxwell ImageD11" before starting python
from ImageD11 import transformer # might require running "module load maxwell ImageD11" before starting python

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"

class GvectorEvaluator:
    
    def __init__(self, directory = None):
        self.directory = None
        self.name = None
        self.ds_eta_omega_file = None
        self.peakIndexer = pyTSXRD.PeakIndexer()
        self.material = {'name': None,
                         'spacegroup': None,
                         'symmetry': None,
                         'unitcell': [None, None, None, None, None, None]}
        self.header = []
        self.dshkls = []
        self.gvectors = []
        self.log = []
        self.merged = []
        self.spot3d_id_reg = 0
        
        self.ds_ranges = []
        self.tth_ranges = []
        self.eta_ranges = []
        self.omega_ranges = []
        self.tth_gap = None
        self.ds_gap = None
        self.eta_gap = None
        self.omega_gap = None
        
        self.ds_tol = 0.005
        self.group_tth_tol = None
        self.group_eta_tol = None
        self.group_omega_tol = None
        
        self.ds_bins = np.zeros((1))
        self.omega_bins = np.zeros((1))
        self.eta_bins = np.zeros((1))
        self.ds_eta_omega = np.zeros((1,1,1))
        self.add_to_log('Initialized GvectorEvaluator object.', False)
        if directory: self.set_attr('directory', directory)
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
        if attr in ['merged', 'header', 'dshkls', 'gvectors']:
            old, new = f'list of {len(old)}', f'list of {len(new)}'
        if attr in ['ds_bins', 'eta_bins', 'omega_bins', 'ds_eta_omega']:
            old, new = f'array of {old.shape}', f'array of {new.shape}'
        self.add_to_log(attr+': '+str(old)+' -> '+str(new))
        return

    
    def add_to_attr(self, attr, value):
        """Method to append value to the choosen attribute. The attribute must be a list."""
        try:
            old_list = getattr(self, attr)
        except:
            old_list = None
        if type(old_list) == list: 
            setattr(self, attr, old_list+[value])
            new_list = getattr(self, attr)
            self.add_to_log(attr+': += '+str(new_list[-1]))
        else:
            raise AttributeError('This attribute is not a list!')
        return    
    
    
    def print(self, also_log = False):
        print(double_separator+'GvectorEvaluator object:')
        print('merged:' , len(self.merged))
        print('spot3d_id_reg:', self.spot3d_id_reg)
        print('directory:', self.directory)
        print('name:'     , self.name)
        print('ds_eta_omega_file:', self.ds_eta_omega_file)
        print('material:' , self.material )
        [print(f'header {i:2}:', r) for i,r in enumerate(self.header)]
        print('dshkls:'  , len(self.dshkls))
        print('gvectors:', len(self.gvectors))
        
        [print(f'ds_range {i}:'   , r) for i,r in enumerate(self.ds_ranges )]
        [print(f'tth_range {i}:'  , r) for i,r in enumerate(self.tth_ranges)]
        [print(f'eta_range {i}:'  , r) for i,r in enumerate(self.eta_ranges)]
        [print(f'omega_range {i}:', r) for i,r in enumerate(self.omega_ranges)]
        print('tth_gap:'  , self.tth_gap)
        print('ds_gap:'   , self.ds_gap)
        print('eta_gap:'  , self.eta_gap)
        print('omega_gap:', self.omega_gap)
        
        print('ds_tol:', self.ds_tol)
        print('group_tth_tol:'  , self.group_tth_tol)
        print('group_eta_tol:'  , self.group_eta_tol)
        print('group_omega_tol:', self.group_omega_tol)
        print('ds_bins:', self.ds_bins.shape, 'array')
        print('eta_bins:', self.eta_bins.shape, 'array')
        print('omega_bins:', self.omega_bins.shape, 'array')
        print('ds_eta_omega:', self.ds_eta_omega.shape, 'array')
        if self.peakIndexer:
            self.peakIndexer.print()
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return
    

    def load_gve(self, directory = None, gve_file = None):
        if directory: self.set_attr('directory', directory)
        if gve_file: self.set_attr('name', gve_file.split('.')[0])
        self.add_to_log('Reading file: '+self.directory+self.name+'.gve', False)
        if not os.path.isfile(self.directory+self.name+'.gve'): raise FileNotFoundError
        material = dict(self.material)
        header, dshkls, gvectors = [], [], []
        
        with open(self.directory+self.name+'.gve', "r") as f:
            for line in f:
                if line[0] == '#':
                    if 'ds h k l' in line: dshkl_keys = line[2:-1].split()
                    elif 'gx  gy  gz' in line or 'xr yr zr' in line:
                        gvector_keys = line[2:-1].split()
                        ind = gvector_keys.index("spot3d_id")
                    else:
                        header.append(line[:-1])
                else:
                    words = line.split()
                    if len(words) == 7 and words[-1] in ['P','I','F', 'A', 'B', 'C','R']:
                        material['unitcell'] = [float(v) for v in words[:6]]
                        material['symmetry'] = words[-1]
                    elif len(words) == 4:
                        d = [float(words[0])] + [int(v) for v in words[1:]]
                        dshkls.append( dict(zip(dshkl_keys,d)) )
                    elif len(words) >= 12:
                        g = [float(v) for v in words]
                        g[ind] = int(words[ind])
                        gvectors.append( dict(zip(gvector_keys,g)) )
        f.close()
        self.set_attr('material', material)
        self.set_attr('header', header)
        self.set_attr('dshkls', dshkls)
        self.set_attr('gvectors', gvectors)
        if 'tth' not in gvector_keys:
            for g in self.gvectors:
                g.update({'tth': self.peakIndexer.geometry.tth_from_ds(g['ds']) })
        #print(f'{len(gvectors)} gvectors loaded.')
        self.add_to_log('File closed!', False)
        return
    
    
    def save_gve(self, directory = None, gve_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if gve_file: self.set_attr('name', gve_file.split('.')[0])
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        
        self.add_to_log('Writing file: '+self.directory+self.name+'.gve', False)
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.name+'.gve'):
            self.add_to_log('File already exist!', False)
            if overwrite:
                self.add_to_log('Overwriting...', False)
                break
            else:
                x = input('Type new name or ! to overwrite, a - abort:')
                if x in ['!']:
                    self.add_to_log('Overwriting...', False)
                    break
                elif x in ['a', 'A']:
                    self.add_to_log('Aborted!', False)
                    return
                else:
                    self.set_attr('name', x.split('.')[0])
        
        f = open(self.directory+self.name+'.gve' ,"w")
        s =[f'{v:.6f}' for v in self.material['unitcell']]
        f.write(' '.join(s+[self.material['symmetry']]) + '\n' )
        for line in self.header: f.write(line + '\n')
        f.write('# ' + ' '.join(self.dshkls[0].keys()) + '\n' )
        for d in self.dshkls:
            s = [f'{v:d}' if type(v)==type(1) else f'{v:.7f}' for v in list(d.values())]
            f.write(' ' + ' '.join(s) + '\n' )
        if len(self.gvectors) > 0:
            f.write('#  ' + '  '.join(self.gvectors[0].keys()) + '\n' )
        for g in self.gvectors:
            s = [f'{v:d}' if type(v)==type(1) else f'{v:.6f}' for v in list(g.values())]
            f.write(' '.join(s) + '\n' )
        f.close()
        self.add_to_log('File closed!', False)
        return
    
    
    def calculate_ranges(self, tth_gap, ds_gap, eta_gap, omega_gap):
        #print(double_separator+'Calculating ranges with [tth, ds, eta, omega] gaps: ')
        #print([tth_gap, ds_gap, eta_gap, omega_gap])
        self.add_to_log('Calculating tth, ds, eta, omega ranges...')
        self.set_attr('tth_gap'  , tth_gap)
        self.set_attr('ds_gap'   , ds_gap)
        self.set_attr('eta_gap'  , eta_gap)
        self.set_attr('omega_gap', omega_gap)
        tt_rs, ds_rs, et_rs, om_rs = [], [], [], []
        for g in self.gvectors:
            try: tt_rs.append( [g['tth']-tth_gap,g['tth']+tth_gap] )
            except: pass
            ds_rs.append( [g['ds']-ds_gap,g['ds']+ds_gap] )
            et_rs.append( [g['eta']-eta_gap,g['eta']+eta_gap] )
            om_rs.append( [g['omega']-omega_gap,g['omega']+omega_gap] )
        try: self.set_attr(  'tth_ranges', merge_overlaps(tt_rs, margin=tth_gap, target=None))
        except: pass
        self.set_attr(   'ds_ranges', merge_overlaps(ds_rs, margin=ds_gap   , target=None))
        self.set_attr(  'eta_ranges', merge_overlaps(et_rs, margin=eta_gap  , target=0))
        self.set_attr('omega_ranges', merge_overlaps(om_rs, margin=omega_gap, target=0))
        return

 
    def remove_not_ranges(self, ds_ranges, tth_ranges, omega_ranges, eta_ranges):
        in_ranges = []
        for i,g in enumerate(self.gvectors):          
            fit = True
            for rng in ds_ranges:
                if g['ds'] < rng[0] or g['ds'] > rng[1]:
                    fit = False
                else:
                    fit = True
                    break
            if not fit: continue

            for rng in tth_ranges:
                if g['tth'] < rng[0] or g['tth'] > rng[1]:
                    fit = False
                else:
                    fit = True
                    break
            if not fit: continue

            for rng in omega_ranges:
                if g['omega'] < rng[0] or g['omega'] > rng[1]:
                    fit = False
                else:
                    fit = True
                    break
            if not fit: continue

            for rng in convert_ranges(eta_ranges):
                if g['eta'] < rng[0] or g['eta'] > rng[1]:
                    fit = False
                else:
                    fit = True
                    break
            if not fit: continue
            else: in_ranges.append(i)
            
        self.add_to_log(f'Removing {len(self.gvectors)-len(in_ranges)} gvectors that are out of ranges. After removal {len(in_ranges)} gvectors left.', False)
        self.set_attr('gvectors', [self.gvectors[i] for i in in_ranges])
        return
    
    def group_gvectors(self, group_y_tol, group_tth_tol, group_eta_tol, group_omega_tol):
        """Groups g-vectors that are close to each other, returns a list of such 'averaged' g-vectors."""
        #print(double_separator+f'Grouping {len(self.gvectors)} g-vectors')
        #print('Tolerances [tth, eta, omega]: ', [group_tth_tol, group_eta_tol, group_omega_tol])
        if not group_tth_tol   >= 0: raise ValueError('group_tth_tol must be non-negative!')
        if not group_eta_tol   >= 0: raise ValueError('group_eta_tol must be non-negative!')
        if not group_omega_tol >= 0: raise ValueError('group_omega_tol must be non-negative!')
        self.add_to_log('Grouping g-vectors...')
        self.set_attr('group_tth_tol', group_tth_tol)
        self.set_attr('group_eta_tol', group_eta_tol)
        self.set_attr('group_omega_tol', group_omega_tol)
    
        tth_list = [g['tth'] for g in self.gvectors]
        chains_0 = group_to_chains(tth_list, group_tth_tol)
        
        chains_1 = []
        for C in chains_0:
            eta_list = [self.gvectors[i]['eta'] for i in C]
            subchains = group_to_chains(eta_list, group_eta_tol)
            tail_to_head = mod_360(eta_list[subchains[-1][-1]] - eta_list[subchains[0][0]], 0)
            if len(subchains)>1 and abs(tail_to_head)<group_eta_tol:
                subchains[0] = subchains[-1] + subchains[0]
                del subchains[-1]
            for c in subchains: chains_1 += [[C[i] for i in c]]
        
        chains_2 = []
        for C in chains_1:
            omega_list = [self.gvectors[i]['omega'] for i in C]
            subchains = group_to_chains(omega_list, group_omega_tol)
            tail_to_head = mod_360(omega_list[subchains[-1][-1]] - omega_list[subchains[0][0]], 0)
            if len(subchains)>1 and abs(tail_to_head)<group_omega_tol:
                subchains[0] = subchains[-1] + subchains[0]
                del subchains[-1]
            for c in subchains: chains_2 += [[C[i] for i in c]]
        
        chains_3 = []
        for C in chains_2:
            eta_list = [self.gvectors[i]['eta'] for i in C]
            subchains = group_to_chains(eta_list, group_eta_tol)               
            tail_to_head = mod_360(eta_list[subchains[-1][-1]] - eta_list[subchains[0][0]], 0)
            if len(subchains)>1 and abs(tail_to_head)<group_eta_tol:
                subchains[0] = subchains[-1] + subchains[0]
                del subchains[-1]
            for c in subchains: chains_3 += [[C[i] for i in c]]

        if group_y_tol == None:
            chains_4 = chains_3
        else:
            chains_4 = []
            for C in chains_3:
                y_list = [self.gvectors[i]['stage_y'] for i in C]
                subchains = group_to_chains(y_list, group_y_tol)
                for c in subchains: chains_4 += [[C[i] for i in c]]
            
        chains_5 = []
        for C in chains_4:
            tth_list = [self.gvectors[i]['tth'] for i in C]
            subchains = group_to_chains(tth_list, group_tth_tol)
            for c in subchains: chains_5 += [[C[i] for i in c]]
        
        av_gvectors = []
        for C in chains_5:
            g_avg = {}
            g_list = [self.gvectors[i] for i in C]
            if 'IMax_int' in g_list[0].keys():
                vals = [g['IMax_int'] for g in g_list] # use intensities for weights
            else :
                vals = [1 for g in g_list] # 
            weights = np.asarray(vals) / sum([v for v in vals])
            for k in g_list[0].keys():
                g_avg[k] = sum([weights[i]*g[k] for i,g in enumerate(g_list)])
            g_avg['spot3d_id'] = g_list[weights.argmax()]['spot3d_id'] # spot3d_id should not be averaged
            av_gvectors.append(g_avg)
            
        ind = np.argsort([g['ds'] for g in av_gvectors]) # Sort in asceding ds 
        self.set_attr('gvectors', [av_gvectors[i] for i in ind])
        return


    def remove_not_inrings(self, ds_tol = None):
        if len(self.gvectors) < 1:
            return
        if ds_tol and ds_tol != 'auto':
            self.set_attr('ds_tol', ds_tol)
        I = indexing.indexer()
        name = self.name
        self.save_gve(gve_file = 'temp.gve', overwrite = True)
        I.readgvfile(self.directory+'temp.gve')
        self.set_attr('name', name)
        subprocess.call('rm '+self.directory+'temp.gve', shell=True)
        
        I.ds_tol = self.ds_tol
        I.assigntorings()
        self.add_to_log(f'Ring assignment using ds_tol = {self.ds_tol}', False)
        for i, tth in enumerate(I.tth): self.gvectors[i].update({'tth':tth})
        in_rings = np.compress(np.greater(I.ra,-1),np.arange(I.gv.shape[0])) # list of indexed peaks
        self.add_to_log(f'Removing {len(self.gvectors)-len(in_rings)} gvectors that are not in rings.', False)
        self.set_attr('gvectors', [self.gvectors[i] for i in in_rings])
        self.save_gve(overwrite = True)
        return
    
    
    def calc_histo(self, omega_pixsize, eta_pixsize, ds_eta_omega_file = None, plot=False, save_arrays = True):
        if ds_eta_omega_file: self.set_attr('ds_eta_omega_file', ds_eta_omega_file)
        self.add_to_log('Calculating hkl_omega and ds_eta_omega histograms...')
        self.add_to_log(f'omega_pixsize  = {omega_pixsize}, eta_pixsize = {eta_pixsize}')
        I = indexing.indexer()
        name = self.name
        self.save_gve(gve_file = 'temp.gve', overwrite = True)
        I.readgvfile(self.directory+'temp.gve')
        self.set_attr('name', name)
        subprocess.call('rm '+self.directory+'temp.gve', shell=True)
        
        I.ds_tol = self.ds_tol
        I.assigntorings()
        self.add_to_log(f'Ring assignment using ds_tol = {self.ds_tol}', False)
        for i, tth in enumerate(I.tth): self.gvectors[i].update({'tth':tth})
        hkl_list = [I.unitcell.ringhkls[ds][-1] for ds in I.unitcell.ringds] # [-1] hkl with highest h (usualy last in the set)
        in_rings = np.compress(np.greater(I.ra,-1),np.arange(I.gv.shape[0])) # list of indexed peaks
        etas = [I.eta[peak] for peak in in_rings]
        omegas = [I.omega[peak] for peak in in_rings]
        ds_bins = np.asarray(I.unitcell.ringds)
        eta_bins = np.linspace(min(etas),max(etas),round(1+(max(etas)-min(etas))/eta_pixsize), dtype=float)
        omega_bins = np.linspace(min(omegas),max(omegas),round(1+(max(omegas)-min(omegas))/omega_pixsize), dtype=float)
        ds_eta_omega = np.zeros( (len(hkl_list),len(eta_bins),len(omega_bins)), dtype=int )
        for peak in in_rings:
            iR = hkl_list.index(I.unitcell.ringhkls[I.unitcell.ringds[I.ra[peak]]][-1])
            t = abs(eta_bins-I.eta[peak])
            iE = np.where(t == np.amin(t))[0][0]
            t = abs(omega_bins-I.omega[peak])
            iO = np.where(t == np.amin(t))[0][0]
            if iR and iE and iO:
                ds_eta_omega[iR,iE,iO] += 1     
        
        self.set_attr('ds_bins'     , ds_bins)
        self.set_attr('omega_bins'  , omega_bins)
        self.set_attr('eta_bins'    , eta_bins)
        self.set_attr('ds_eta_omega', ds_eta_omega)
        
        if save_arrays:
            N1, N2, N3 = ds_eta_omega.shape
            self.set_attr('ds_eta_omega_file', self.name+f'_omega_eta_hkl_{N3}x{N2}x{N1}.raw')
            output_file = open(self.directory+self.ds_eta_omega_file, 'wb')
            np.float32(ds_eta_omega).tofile(output_file)
            output_file.close()
            self.add_to_log('Saved histogram: '+self.ds_eta_omega_file, False)
        if plot:
            fig = plt.figure(figsize=(8, 6))
            sub1 = fig.add_subplot(131)
            x1 = [g['omega'] for g in self.gvectors]
            y1 = [g['eta'  ] for g in self.gvectors]
            plt.scatter(x1, y1, s = 1)
            plt.title(f'Scatter plot of all {len(self.gvectors)} gvectors')
            plt.xlabel('omega (deg)')
            plt.ylabel('eta (deg)')

            sub2 = fig.add_subplot(132)
            x1 = [g['ds'   ] for g in self.gvectors]
            y1 = [g['omega'] for g in self.gvectors]
            plt.scatter(x1, y1, s = 1)
            plt.title(f'Scatter plot of all {len(self.gvectors)} gvectors')
            plt.xlabel('ds')
            plt.ylabel('omega (deg)')

            sub3 = fig.add_subplot(133)
            x1 = [g['tth'] for g in self.gvectors]
            y1 = [g['eta'] for g in self.gvectors]
            plt.scatter(x1, y1, s = 1)
            plt.title(f'Scatter plot of all {len(self.gvectors)} gvectors')
            plt.xlabel('tth (deg)')
            plt.ylabel('eta (deg)')
            plt.show()
            fig.savefig(self.directory+self.name+'_scatter.png')
            self.add_to_log('Saved projections: '+self.name+'_scatter.png', False)
        
        n_peaks = self.ds_eta_omega.sum(2).sum(1) 
        #for i,ds in enumerate(self.ds_bins): print(f'ds, n_peaks:  {ds:.3f}, {n_peaks[i]:d}')
        
        if plot:
            x1 = []
            x2 = []
            for g in self.gvectors:
                nearest_ds  = -1000000
                nearest_tth = -1000000
                for iR,ds in enumerate(I.unitcell.ringds):
                    tth = 2*np.degrees(np.arcsin(ds*I.wavelength/2))
                    if abs(ds-g['ds']) < abs(nearest_ds-g['ds']):
                        r = iR
                        nearest_ds  = ds
                        nearest_tth = tth
                x1.append(g['ds' ]-nearest_ds  + r*0.01)
                x2.append(g['tth']-nearest_tth + r*0.5)

            fig = plt.figure(figsize=(8, 6))
            
            y1 = [g['eta'  ] for g in self.gvectors]
            sub1 = fig.add_subplot(121)
            plt.scatter(x1, y1, s = 1)
            plt.title(f'Scatter plot of all {len(self.gvectors)} gvectors')
            plt.xlabel('ds-ds_hkl')
            plt.ylabel('eta (deg)')
            plt.xlim([-0.01, r*0.01+0.01])
            
            y2 = [g['omega'  ] for g in self.gvectors]
            sub1 = fig.add_subplot(122)
            plt.scatter(x2, y2, s = 1)
            plt.title(f'Scatter plot of all {len(self.gvectors)} gvectors')
            plt.xlabel('tth-tth_hkl')
            plt.ylabel('omega (deg)')
            plt.xlim([-0.5, r*0.5+0.01])
            plt.show()
            fig.savefig(self.directory+self.name+'_scatter_delta_ds.png')
            self.add_to_log('Saved eta_delta_ds: '+self.name+'_scatter_eta_delta_ds.png', False)

        return
    
    
    def trace_peak_by_spot3d_id(self, spot3d_id):
        if len(self.merged) < 1:
            return self, [p for p in self.peakIndexer.peaks if p['spot3d_id'] == spot3d_id][0]
        else:
            GE = self.merged[0]
            for ge in self.merged:
                if ge.spot3d_id_reg <= spot3d_id:
                    if ge.spot3d_id_reg >= GE.spot3d_id_reg:
                        GE = ge
            spot3d_id = spot3d_id - GE.spot3d_id_reg
            if len(GE.merged) < 1:
                return GE, [p for p in GE.peakIndexer.peaks if p['spot3d_id'] == spot3d_id][0]
            else:
                return GE.trace_peak_by_spot3d_id(spot3d_id)

    
def merge_GE_list(directory, name, GE_list, spot3d_id_reg):
    self = GvectorEvaluator(directory)
    self.set_attr('name', name)
    
    material = dict(GE_list[0].material)
    header = []
    dshkls = []
    for GE in GE_list:
        if len(GE.header) > len(header): header = 1*GE.header
        if len(GE.dshkls) > len(dshkls): dshkls = 1*GE.dshkls
        if GE.material['symmetry'] != material['symmetry']:
            raise ValueError('Symmetries are different!')
        if GE.material['unitcell'] != material['unitcell']:
            raise ValueError('Unit cells are different!')
        if GE.material['spacegroup'] != material['spacegroup']:
            raise ValueError('Spacegroups are different!')
        max_spot_3d_id = max([g['spot3d_id'] for g in GE.gvectors])
        if spot3d_id_reg <=  max_spot_3d_id:
            raise ValueError(f'Provided spot3d_id_reg ({spot3d_id_reg}) is too small. At least {max_spot_3d_id} needed.')

    self.set_attr('material', material)
    self.set_attr('header', header)
    self.set_attr('dshkls', dshkls)

    common_keys = []
    for GE in GE_list:
        common_keys += [k for k in GE.gvectors[0].keys() if k not in common_keys]
    gvectors = []
    for iGE, GE in enumerate(GE_list):
        self.add_to_attr('merged', copy.copy(GE))
        self.merged[-1].spot3d_id_reg = spot3d_id_reg*iGE # 
        for g in GE.gvectors:
            c = {key:value for key,value in g.items() if key in common_keys}
            c['spot3d_id'] += spot3d_id_reg*iGE
            if 'stage_y' not in common_keys:
                try:
                    stage_pos = GE.peakIndexer.sweepProcessor.position
                except:
                    stage_pos = GE.merged[0].peakIndexer.sweepProcessor.position
                c['stage_x'], c['stage_y'], c['stage_z'] = stage_pos
            gvectors.append(c)           
                    
    ind = np.argsort([g['ds'] for g in gvectors]) # Sort in asceding ds 
    self.set_attr('gvectors', [gvectors[i] for i in ind])
    self.add_to_log(f'Combined {len(GE_list)} parent GvectorEvaluator objects into a new one.', False)
    return self