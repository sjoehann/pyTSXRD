# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 16:00:49 2022

@author: sjö

Class to run grainspotter.
"""

import os, subprocess
from datetime import datetime
from pyTSXRD.angles_and_ranges import merge_overlaps
single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"
        
class GrainSpotter:
    
    def __init__(self, directory = None):
        self.log = []
        self.directory = None
        self.ini_file = None
        self.gve_file = None
        self.log_file = None
        self.spacegroup = None
        self.ds_ranges = []
        self.tth_ranges = []
        self.eta_ranges = []
        self.omega_ranges = []
        self.domega = None
        self.cuts = [] # [min_measuments, min_completeness, min_uniqueness]
        self.eulerstep = None # [stepsize] : angle step size in Euler space
        self.uncertainties = [] # [sigma_tth sigma_eta sigma_omega] in degrees
        self.nsigmas = None # [Nsig] : maximal deviation in sigmas
        self.Nhkls_in_indexing = None # [Nfamilies] : use first Nfamilies in indexing
        self.random = None # [Npoints] random sampling of orientation space trying Npoints sample points
        self.positionfit = None # True/False fit the position of the grain
        self.minfracg = None # True/False stop search when minfracg (0..1) of gvectors assigned to grains
        self.genhkl = None # True/Falsegenerate list of reflections
        self.add_to_log('Initialized GrainSpotter object.', False)
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
        self.add_to_log(attr+': '+str(old)+' -> '+str(new),False)
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
        print(double_separator+'GrainSpotter object:')
        print('directory:', self.directory)
        print('ini_file:', self.ini_file)
        print('gve_file:', self.gve_file)
        print('log_file:', self.log_file)
        [print(  f'ds_range {i}:' , r) for i,r in enumerate(self.ds_ranges   )]
        [print(  f'tth_range {i}:', r) for i,r in enumerate(self.tth_ranges  )]
        [print(  f'eta_range {i}:', r) for i,r in enumerate(self.eta_ranges  )]
        [print(f'omega_range {i}:', r) for i,r in enumerate(self.omega_ranges)]
        print('domega:', self.domega)
        print('gve_file:', self.gve_file)
        print('cuts:', self.cuts)
        print('eulerstep:', self.eulerstep)
        print('uncertainties:', self.uncertainties)
        print('nsigmas:', self.nsigmas)
        print('Nhkls_in_indexing:', self.Nhkls_in_indexing)
        print('random:', self.random)
        print('positionfit:', self.positionfit)
        print('minfracg: ', self.minfracg)
        print('genhkl:', self.genhkl)
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return
    

    def load_ini(self, directory = None, ini_file = None):
        if directory: self.set_attr('directory', directory)
        if ini_file: self.set_attr('ini_file', ini_file)
        self.add_to_log(f'Reading file: {self.directory+self.ini_file}', False)
        if not os.path.isfile(self.directory+self.ini_file): raise FileNotFoundError
        self.set_attr('ds_ranges', [])
        self.set_attr('tth_ranges', [])
        self.set_attr('eta_ranges', [])
        self.set_attr('omega_ranges', [])
        with open(self.directory+self.ini_file, "r") as f:
            for line in f:
                if line[0] == '#' or len(line) < 2: continue
                words = line.split() # words in line
                if ('spacegroup' in words[0]):
                    self.set_attr('spacegroup', int(words[1]))
                elif ('dsrange' in words[0]):
                    if words[0][0] == '!': self.set_attr('ds_ranges', [])
                    else: self.add_to_attr('ds_ranges', [float(v) for v in words[1:3]])
                elif ('tthrange' in words[0]):
                    if words[0][0] == '!': self.set_attr('tth_ranges', [])
                    else: self.add_to_attr('tth_ranges', [float(v) for v in words[1:3]])
                elif ('etarange' in words[0]):
                    if words[0][0] == '!': self.set_attr('eta_ranges', [])
                    else:
                        eta_ranges = merge_overlaps( [float(v) for v in words[1:3]], margin=0, target=0)
                        self.add_to_attr('eta_ranges', eta_ranges[0])
                elif ('omegarange' in words[0]):
                    if words[0][0] == '!': self.set_attr('omega_ranges', [])
                    else: self.add_to_attr('omega_ranges', [float(v) for v in words[1:3]])
                elif ('domega' in words[0]):
                    self.set_attr('domega', float(words[1]))
                elif ('filespecs' in words[0]):
                    self.set_attr('gve_file', words[1].rsplit('/', 1)[-1])
                    self.set_attr('log_file', words[2].rsplit('/', 1)[-1])
                elif ('cuts' in words[0]):
                    self.set_attr('cuts', [int(float(words[1])), float(words[2]), float(words[3])])
                elif ('uncertainties' in words[0]):
                    self.set_attr('uncertainties', [float(v) for v in words[1:4]])
                elif ('random' in words[0]):
                    self.set_attr('random', int(float(words[1])))
                elif ('eulerstep' in words[0]):
                    self.set_attr('eulerstep', float(words[1]))
                elif ('nsigmas' in words[0]):
                    self.set_attr('nsigmas', float(words[1]))
                elif ('minfracg' in words[0]):
                    if words[0][0] == '!': self.set_attr('minfracg', None)
                    else: self.set_attr('minfracg', float(words[1]))
                elif ('Nhkls_in_indexing' in words[0]):
                    if words[0][0] == '!': self.set_attr('Nhkls_in_indexing', None)
                    else: self.set_attr('Nhkls_in_indexing', int(float(words[1])))
                elif ('genhkl' in words[0]):
                    if words[0][0] == '!': self.set_attr('genhkl', False)
                    else: self.set_attr('genhkl', True)
                elif ('positionfit' in words[0]):
                    if words[0][0] == '!': self.set_attr('positionfit', False)
                    else: self.set_attr('positionfit', True)
        f.close()
        self.add_to_log('File closed!', False)
        return
 
        
    def save_ini(self, directory = None, ini_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if ini_file: self.set_attr('ini_file', ini_file)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        self.add_to_log(f'Writing file: {self.directory+self.ini_file}', False)
        
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.ini_file):
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
                    self.set_attr('ini_file', x)
        
        f = open(self.directory+self.ini_file ,"w")
        
        f.write('spacegroup {}           '.format(self.spacegroup))
        f.write('!# spacegroup [space group nr]\n')
        
        c = '!# dsrange  [min max], reciprocal d-spacing range, few ranges can be specified\n'
        if self.ds_ranges:
            for r in self.ds_ranges:
                f.write('dsrange {:0.2f} {:0.2f}         '.format(r[0],r[1]) + c)
        else:
            f.write('!dsrange 0.2 0.5         ' + c)
        
        c = '!# tthrange [min max], few ranges can be specified\n'
        if self.tth_ranges:
            for r in self.tth_ranges:
                f.write('tthrange {:0.2f} {:0.2f}       '.format(r[0],r[1]) + c)
        else:
            f.write('!tthrange 0 30           ' + c)

        c = '!# etarange [min max], from 0 to 360, few ranges can be specified\n'
        if self.eta_ranges:
            for r in merge_overlaps(self.eta_ranges, margin=0, target=180):
                f.write('etarange {:0.1f} {:0.1f}         '.format(r[0],r[1]) + c)
        else:
            f.write('!etarange 0 360          ' + c)
            
        c = '!# omegarange [min max], from -180 to 180, few ranges can be specified\n'
        if self.omega_ranges:
            for r in self.omega_ranges:
                f.write('omegarange {:0.1f} {:0.1f}   '.format(r[0],r[1]) + c)
        else:
            f.write('!omegarange -180 180     ' + c)
        
        f.write('domega {}              '.format(self.domega))
        f.write('!# domega [stepsize] in omega, degrees\n')
        
        f.write('filespecs {} {} '.format(self.directory+self.gve_file,self.directory+self.log_file))
        f.write('!# filespecs [gvecsfile grainsfile]\n')
        
        v = self.cuts
        f.write('cuts {} {} {}        '.format(v[0],v[1],v[2]))
        f.write('!# cuts [min_measuments min_completeness min_uniqueness]\n')
        
        f.write('eulerstep {}              '.format(self.eulerstep))
        f.write('!# eulerstep [stepsize] : angle step size in Euler space\n')
        
        v = self.uncertainties
        f.write('uncertainties {} {} {} '.format(v[0],v[1],v[2]))
        f.write('!# uncertainties [sigma_tth sigma_eta sigma_omega] in degrees\n')
        
        f.write('nsigmas {}              '.format(self.nsigmas))
        f.write('!# nsigmas [Nsig] : maximal deviation in sigmas\n')
        
        n = self.Nhkls_in_indexing
        if n: f.write( 'Nhkls_in_indexing {}      '.format(n))
        else: f.write( '!Nhkls_in_indexing 15    ')
        f.write('!# Nhkls_in_indexing [Nfamilies] : use first Nfamilies in indexing\n')
        
        if self.random: f.write('random {}             '.format(self.random))
        else:           f.write('!random 10000            ')
        f.write('!# random sampling of orientation space trying 10000 sample points\n')
        
        if self.positionfit: f.write('positionfit              ')
        else:                f.write('!positionfit             ')
        f.write('!# fit the position of the grain\n')
        
        if self.minfracg: f.write('minfracg {}              '.format(self.minfracg))
        else:             f.write('!minfracg 0.2            ')
        f.write('!# stop search when minfracg (0..1) of the gvectors have been assigned to grains\n')
            
        if self.genhkl: f.write('genhkl                   ')
        else:           f.write('!genhkl                  ')
        f.write('!# generate list of reflections\n')
        
        f.write('# min_measuments: grain chosen if >= this amount of peaks per grain present\n')
        f.write('# min_completeness: grain chosen if > this fraction of the expected peaks present\n')
        f.write('# min_uniqueness: no idea, just put any number\n')
                    
        f.close()
        self.add_to_log('File closed!', False)
        return
        
        
    def run_grainspotter(self, directory = None, ini_file = None, gve_file = None, log_file = None):
        if directory: self.set_attr('directory', directory)
        if ini_file : self.set_attr('ini_file' , ini_file)
        if gve_file : self.set_attr('gve_file' , gve_file)
        if log_file : self.set_attr('log_file' , log_file)
        self.save_ini(overwrite = True)
        
        del_old  = subprocess.call('rm '+self.directory+self.log_file, shell=True)
        del_old += subprocess.call('rm '+self.directory+self.log_file.replace('.log','.ubi'), shell=True)
        del_old += subprocess.call('rm '+self.directory+self.log_file.replace('.log','.gff'), shell=True)
        
        if del_old > 0: self.add_to_log(f"Deleted old .log .gff .ubi files.", False)
        self.add_to_log(f'Running grainspotter on: {self.directory}{self.gve_file}', True)
        self.add_to_log(f'Using ini_file: {self.directory}{self.ini_file}', True)

        command = 'grainspotter ' + self.directory+self.ini_file
        process = subprocess.run(command.split(), check=False,stdout=subprocess.PIPE, universal_newlines=True, cwd=self.directory)
        self.add_to_log('Output:'+process.stdout, True)
        #print('Last line in the output:'+process.stdout.splitlines()[-1])
        return
