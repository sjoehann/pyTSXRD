# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 11:00:49 2022

@author: sjö

Class to work with a list of peaks and *.flt files.
"""

import os, copy
import numpy as np
from datetime import datetime
from ImageD11 import indexing
from ImageD11 import transformer
import pyTSXRD

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"

class PeakIndexer:
    
    def __init__(self, directory = None):
        self.directory = None
        self.name = None
        self.geometry = pyTSXRD.Geometry()
        self.header  = []
        self.peaks = []
        self.log = []
        self.sweepProcessor = pyTSXRD.SweepProcessor
        self.add_to_log('Initialized PeakIndexer object.', False)
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
        if attr in ['absorbed', 'header', 'peaks']:
            old, new = f'list of {len(old)}', f'list of {len(new)}'
        if attr == 'geometry' and old is not None: old = old.__dict__
        if attr == 'geometry' and new is not None: new = new.__dict__       
        if attr == 'sweepProcessor' and old is not None: old = old.__dict__
        if attr == 'sweepProcessor' and new is not None: new = new.__dict__ 
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
        print(double_separator+'PeakIndexer object:')
        print('directory:', self.directory)
        print('name:'     , self.name )
        [print(f'header {i:2}:', r) for i,r in enumerate(self.header)]
        print('peaks:', len(self.peaks))
        try:
            self.geometry.print()
        except:
            print("Geometry is missing!")
        try:
            self.sweepProcessor.print()
        except:
            print("sweepProcessor is missing!")
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return
    
    
    def load_flt(self, directory = None, flt_file = None):
        if directory: self.set_attr('directory', directory)
        if flt_file: self.set_attr('name', flt_file.replace('.flt', '') )
        self.add_to_log('Reading file:'+self.directory+self.name+'.flt', False)
        if not os.path.isfile(self.directory+self.name+'.flt'): raise FileNotFoundError  
        header, peaks, inds = [], [], []
        with open(self.directory+self.name+'.flt', "r") as f:
            for line in f:
                if line[0] == '#':
                    if 'sc  fc  omega' in line:
                        peak_keys = line[2:-1].split()
                        inds.append(peak_keys.index("Number_of_pixels"))
                        inds.append(peak_keys.index("IMax_s"))
                        inds.append(peak_keys.index("IMax_f"))
                        inds.append(peak_keys.index("Min_s"))
                        inds.append(peak_keys.index("Max_s"))
                        inds.append(peak_keys.index("Min_f"))
                        inds.append(peak_keys.index("Max_f"))
                        inds.append(peak_keys.index("onfirst"))
                        inds.append(peak_keys.index("onlast"))
                        inds.append(peak_keys.index("spot3d_id"))
                    else: header.append(line[:-1])
                else:
                    words = line.split()
                    if len(words) == len(peak_keys):
                        p = [float(v) for v in words]
                        for i in inds: p[i] = int(words[i])
                        peaks.append( dict(zip(peak_keys,p)) )
        f.close()
        self.set_attr('header', header)
        self.set_attr('peaks', peaks)
        #print(f'{len(peaks)} peaks loaded.')
        self.add_to_log('File closed!', False)
        return
 

    def save_flt(self, directory = None, flt_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if flt_file: self.set_attr('name', flt_file.split('.')[0])
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        
        self.add_to_log('Writing file: '+self.directory+self.name+'.flt', False)
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.name+'.flt'):
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
        
        f = open(self.directory+self.name+'.flt' ,"w") 
        for line in self.header: f.write(line + '\n')
        if len(self.peaks)>0: f.write('#  ' + '  '.join(self.peaks[0].keys()) + '\n' )
        for p in self.peaks:
            s = [f'{v:d}' if type(v)==type(1) else f'{v:.4f}' for v in list(p.values())]
            f.write('  ' + '  '.join(s) + '\n' )
        f.close()
        self.add_to_log('File closed!', False)
        return


    def run_indexer(self):
        self.save_flt(overwrite = True)
        self.geometry.save_par(directory = self.directory, par_file = self.name+'.par', overwrite = True)
        self.geometry.save_yml(directory = self.directory, yml_file = self.name+'.yml', overwrite = True)
        self.geometry.save_poni(directory = self.directory, poni_file = self.name+'.poni', overwrite = True)
        self.add_to_log('Running indexer on: '+self.directory+self.name+'.flt', False)
        self.add_to_log('Using par_file: '+self.directory+self.name+'.par', False)
        obj = transformer.transformer()
        obj.loadfiltered( self.directory+self.name+'.flt' )
        obj.loadfileparameters( self.directory+self.name+'.par' )
        obj.compute_tth_eta( )
        obj.addcellpeaks( )
        obj.computegv( )
        obj.savegv( self.directory + self.name+'.gve' )
        self.add_to_log('Resulted gvectors saved in: '+self.directory+self.name+'.gve', False)
        return
    
    
    def generate_GvectorEvaluator(self, directory = None, name = None):
        if not directory: directory = self.directory
        if not name: name = self.name
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.add_to_log('Created directory: '+directory, False)
        GE = pyTSXRD.GvectorEvaluator(directory = directory)
        GE.set_attr('peakIndexer', self)
        GE.set_attr('material', copy.copy(self.geometry.material))
        GE.load_gve(gve_file = name+'.gve') # merged
        for gv in GE.gvectors:
            gv['stage_x'] = self.sweepProcessor.position[0]
            gv['stage_y'] = self.sweepProcessor.position[1]
            gv['stage_z'] = self.sweepProcessor.position[2]
                
        self.add_to_log('Generated GvectorEvaluator in '+directory+' named '+name, False)
        return GE