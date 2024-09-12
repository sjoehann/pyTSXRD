# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:00:49 2022

@author: sjö

Class to handle diffraction geometry in grain-resolved 3D XRD data analysis.
Based on ImageD11 conventions (*.par files).
Allows input/output from files based on HEXRD (*.yml) and pyFAI (*.poni).
Can be used to average multiple geometry objects (different fits, for example).
Runs indexer.
"""

import os, yaml
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"


class Geometry:
    
    def __init__(self, directory = None):
        self.log = []
        self.directory = None
        self.name = None
        self.material = {'name': None,
                         'spacegroup': None,
                         'symmetry': None,
                         'unitcell': [None, None, None, None, None, None]}
        self.chi       = None
        self.distance  = None
        self.buffer    = None
        self.saturation_level = None
        self.fit_tolerance = None
        self.min_bin_prob  = None
        self.no_bins   = None
        self.O         = [[None, None], [None, None]]
        self.omegasign = None
        self.t     = [0,0,0]
        self.tilt  = [None, None, None]
        self.wedge = None
        self.weight_hist_intensities = None
        self.wavelength = None
        self.y_center   = None
        self.y_size     = None
        self.dety_size  = None
        self.detz_size  = None
        self.z_center   = None
        self.z_size     = None
        self.spline_file= None
        self.add_to_log('Initialized Geometry object.', False)
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
        print(double_separator+'Geometry object:')
        print('directory:', self.directory)
        print('name:'     , self.name )
        print('material:' , self.material )
        print('chi:'      , self.chi)
        print('distance:' , self.distance)
        print('buffer:'   , self.buffer)
        print('saturation_level:', self.saturation_level)
        print('fit_tolerance:', self.fit_tolerance)
        print('min_bin_prob:' , self.min_bin_prob )
        print('no_bins:'  , self.no_bins)
        print('O: '       , self.O)
        print('omegasign:', self.omegasign)
        print('t:'        , self.t)
        print('tilt:'     , self.tilt )
        print('wedge:'    , self.wedge)
        print('weight_hist_intensities:', self.weight_hist_intensities)
        print('wavelength:', self.wavelength)
        print(f'(y_size, z_size): ({self.y_size}, {self.z_size})')
        print(f'(dety_size, detz_size): ({self.dety_size}, {self.detz_size})')
        print(f'(y_center, z_center): ({self.y_center}, {self.z_center})')
        print('spline_file:', self.spline_file)
        
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return

    
    def load_par(self, directory = None, par_file = None):
        if directory: self.set_attr('directory', directory)
        if par_file : self.set_attr('name', par_file.replace('.par', ''))
        self.add_to_log(f'Reading file: {self.directory+self.name}.par', False)
        if not os.path.isfile(self.directory+self.name+'.par'): raise FileNotFoundError
        
        m  = self.material.copy()
        O = [[None, None], [None, None]]
        t = [None, None, None]
        tilt = [None, None, None]
        with open(self.directory+self.name+'.par', "r") as f:
            for line in f:
                if line[0] == '#' or len(line) < 2: continue
                words = line.split() # words in line
                if words[1] in ['None','none', 'nan']: continue
                if   ('cell__a'    == words[0]): m['unitcell'][0] = float(words[1])
                elif ('cell__b'    == words[0]): m['unitcell'][1] = float(words[1])
                elif ('cell__c'    == words[0]): m['unitcell'][2] = float(words[1])
                elif ('cell_alpha' == words[0]): m['unitcell'][3] = float(words[1])
                elif ('cell_beta'  == words[0]): m['unitcell'][4] = float(words[1])
                elif ('cell_gamma' == words[0]): m['unitcell'][5] = float(words[1])
                elif ('cell_lattice' in words[0]): m['symmetry'] = words[1]
                elif ('chi' == words[0]): self.set_attr('chi', float(words[1]))
                elif ('distance' == words[0]): self.set_attr('distance', float(words[1]))
                elif ('fit_tolerance' == words[0]):
                    self.set_attr('fit_tolerance', float(words[1]))
                elif ('min_bin_prob' == words[0]):
                    self.set_attr('min_bin_prob', float(words[1]))
                elif ('no_bins' == words[0]): self.set_attr('no_bins', int(words[1]))
                elif ('o11' == words[0]): O[0][0] = int(words[1])
                elif ('o12' == words[0]): O[0][1] = int(words[1])
                elif ('o21' == words[0]): O[1][0] = int(words[1])
                elif ('o22' == words[0]): O[1][1] = int(words[1])
                elif ('omegasign' == words[0]):
                    self.set_attr('omegasign', int(float(words[1])))
                elif ('tilt_x' == words[0]): tilt[0] = float(words[1])
                elif ('tilt_y' == words[0]): tilt[1] = float(words[1])
                elif ('tilt_z' == words[0]): tilt[2] = float(words[1])
                elif ('t_x'    == words[0]): t[0]    = float(words[1])
                elif ('t_y'    == words[0]): t[1]    = float(words[1])
                elif ('t_z'    == words[0]): t[2]    = float(words[1])
                elif ('wedge'  == words[0]): self.set_attr('wedge', float(words[1]))
                elif ('weight_hist_intensities' == words[0]):
                    if words[1] in ['False', 'None']: self.set_attr('weight_hist_intensities', False)
                    else: self.set_attr('weight_hist_intensities', int(words[1]))
                elif ('wavelength' == words[0]):
                    self.set_attr('wavelength', float(words[1]))
                elif ('y_center'   == words[0]): self.set_attr('y_center' , float(words[1]))
                elif ('y_size'     == words[0]): self.set_attr('y_size'   , float(words[1]))
                elif ('dety_size'  == words[0]): self.set_attr('dety_size', int(words[1]))
                elif ('z_center'   == words[0]): self.set_attr('z_center' , float(words[1]))
                elif ('z_size'     == words[0]): self.set_attr('z_size'   , float(words[1]))
                elif ('detz_size'  == words[0]): self.set_attr('detz_size', int(words[1]))
                elif ('spline_file'== words[0]): self.set_attr('spline_file', words[1])
            self.set_attr('material', m)
            self.set_attr('t', t)
            self.set_attr('tilt', tilt)
        f.close()
        self.add_to_log('File closed!', False)
        
        if None in np.asarray(self.O):
            while None in np.asarray(O):
                print('Image orientation (O-matrix) is missing!')
                x = input('Set it as 4 spaced numbers (O11 O12 O21 O22):').split()
                O = [[int(v) for v in x[0:2]], [int(v) for v in x[2:]]]
        self.set_attr('O', O)
            
        return
 
        
    def save_par(self, directory = None, par_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if par_file : self.set_attr('name', par_file.replace('.par', ''))
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        self.add_to_log(f'Writing file: {self.directory+self.name}.par', False)
        
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.name+'.par'):
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
                    self.set_attr('name', x.replace('.par', ''))
        
        f = open(self.directory+self.name+'.par' ,"w")
        f.write( 'cell__a {}\n'.format(self.material['unitcell'][0]) )
        f.write( 'cell__b {}\n'.format(self.material['unitcell'][1]) )
        f.write( 'cell__c {}\n'.format(self.material['unitcell'][2]) )
        f.write( 'cell_alpha {}\n'.format(self.material['unitcell'][3]) )
        f.write( 'cell_beta {}\n'.format(self.material['unitcell'][4]) )
        f.write( 'cell_gamma {}\n'.format(self.material['unitcell'][5]) )
        f.write( 'cell_lattice_[P,A,B,C,I,F,R] {}\n'.format(self.material['symmetry']) )
        f.write( 'chi {}\n'.format(self.chi) )
        f.write( 'distance {:0.3f}\n'.format(self.distance) )
        f.write( 'fit_tolerance {}\n'.format(self.fit_tolerance) )
        if self.min_bin_prob: f.write( 'min_bin_prob {}\n'.format(self.min_bin_prob) )
        if self.no_bins: f.write( 'no_bins {}\n'.format(self.no_bins) )
        f.write( 'o11 {}\n'.format(self.O[0][0]) )
        f.write( 'o12 {}\n'.format(self.O[0][1]) )
        f.write( 'o21 {}\n'.format(self.O[1][0]) )
        f.write( 'o22 {}\n'.format(self.O[1][1]) )
        f.write( 'omegasign {}\n'.format(self.omegasign) )
        f.write( 't_x {}\n'.format(self.t[0]) )
        f.write( 't_y {}\n'.format(self.t[1]) )
        f.write( 't_z {}\n'.format(self.t[2]) )
        f.write( 'tilt_x {:0.6f}\n'.format(self.tilt[0]) )
        f.write( 'tilt_y {:0.6f}\n'.format(self.tilt[1]) )
        f.write( 'tilt_z {:0.6f}\n'.format(self.tilt[2]) )
        f.write( 'wavelength {:0.6f}\n'.format(self.wavelength) )
        f.write( 'wedge {}\n'.format(self.wedge) )
        if self.weight_hist_intensities: f.write( 'weight_hist_intensities {}\n'.format(self.weight_hist_intensities) )
        f.write( 'y_center {:0.3f}\n'.format(self.y_center) )
        f.write( 'y_size {}\n'.format(self.y_size) )
        if self.dety_size: f.write( 'dety_size {}\n'.format(self.dety_size) )
        f.write( 'z_center {:0.3f}\n'.format(self.z_center) )
        f.write( 'z_size {}\n'.format(self.z_size) )
        if self.detz_size: f.write( 'detz_size {}\n'.format(self.detz_size) )
        if self.spline_file: f.write( 'spline_file {}\n'.format(self.spline_file) )
        f.close()
        self.add_to_log('File closed!', False)
        return
    
    
    def into_hexrd_definitions(self, det_num=1):
        if None in np.asarray(self.O):
            O = [[None, None], [None, None]]
            while None in np.asarray(O):
                print('Image orientation (O-matrix) is missing!')
                x = input('Set it as 4 spaced numbers (O11 O12 O21 O22):').split()
                O = [[int(v) for v in x[0:2]], [int(v) for v in x[2:]]]
            self.set_attr('O', O)
            
        ### HEXRD[x,y,z] = FABLE[-y,z,-x] ###
        r0 = Rotation.from_euler('z', -self.tilt[0])  # 1st is about FABLE:+x (radians) = about HEXRD:-z
        r1 = Rotation.from_euler('x', -self.tilt[1])  # 2nd is about FABLE:+y (radians) = about HEXRD:-x
        r2 = Rotation.from_euler('y',  self.tilt[2])  # 3rd is about FABLE:+z (radians) = about HEXRD:+y
        hexrd_tilt = (r0*r1*r2).as_euler('xyz') # in radians
        
        det_imgsize_sf = np.asarray([self.detz_size  , self.dety_size  ])   # (det_slow, det_fast)
        det_pixsize_sf = np.asarray([self.z_size     , self.y_size     ])   # in um here
        det_beampos_sf = np.asarray([self.z_center   , self.y_center   ])   # in pixels
        det_centpos_sf = np.asarray([self.detz_size-1, self.dety_size-1])/2 # in pixels

        det_beampos_sf = (det_beampos_sf-det_centpos_sf)*det_pixsize_sf   # in um from img. center
        
        fable_imgsize_zy = np.matmul(np.asarray(self.O), det_imgsize_sf) # rot. about img. center
        fable_pixsize_zy = np.matmul(np.asarray(self.O), det_pixsize_sf) # rot. about img. center
        fable_beampos_zy = np.matmul(np.asarray(self.O), det_beampos_sf) # rot. about img. center

        hexrd_imgsize = [round(abs(fable_imgsize_zy[0])), round(abs(fable_imgsize_zy[1]))]
        hexrd_pixsize = [      abs(fable_pixsize_zy[0]) ,       abs(fable_pixsize_zy[1]) ]
        
        hexrd_centpos_befor_rot = np.asarray( [fable_beampos_zy[1], -fable_beampos_zy[0], 0] )
        hexrd_centpos_after_rot = (r0*r1*r2).apply(hexrd_centpos_befor_rot)
        hexrd_translt = hexrd_centpos_after_rot + np.asarray([0, 0, -self.distance])
        
        if   self.O == [[-1, 0], [ 0,-1]]: hexrd_orientation = 'none'
        elif self.O == [[ 0,-1], [-1, 0]]: hexrd_orientation = 't'
        elif self.O == [[ 1, 0], [ 0,-1]]: hexrd_orientation = 'v'
        elif self.O == [[-1, 0], [ 0, 1]]: hexrd_orientation = 'h'
        elif self.O == [[ 1, 0], [ 0, 1]]: hexrd_orientation = 'r180'
        elif self.O == [[ 0, 1], [-1, 0]]: hexrd_orientation = 'r90'
        elif self.O == [[ 0,-1], [ 1, 0]]: hexrd_orientation = 'r270'
        else                             : hexrd_orientation = 'unknown'
        
        energy = 12.39842/self.wavelength
        azimuth = 90
        if self.wedge: polar_angle = 90.0-np.degrees(self.wedge) # must be checked before using!
        else         : polar_angle = 90.0
        if self.chi: chi = np.degrees(self.chi) # must be checked before using!
        else       : chi = 0
        
        saturation_level = self.saturation_level
        buffer = self.buffer if self.buffer else 0

        pars = {'id': 'instrument',
                'beam':{
                    'energy': energy,
                    'vector':{
                        'azimuth'    : 90.0,   # must be checked before using!
                        'polar_angle': polar_angle}}, # must be checked before using!
                'oscillation_stage':{
                    'chi': chi, 
                    'translation': [0, 0, 0]}, # in FABLE t is the translation of a grain (not of the stage)
                'detectors':{
                    f'detector_{det_num}':{
                        'transform':{
                            'tilt': [float(v) for v in hexrd_tilt],
                            'translation': [float(v/1000) for v in hexrd_translt],
                            'orientation': hexrd_orientation},
                        'pixels':{
                            'rows'   : int(hexrd_imgsize[0]),
                            'columns': int(hexrd_imgsize[1]),
                            'size'   : [float(v/1000) for v in hexrd_pixsize]},
                        'saturation_level': saturation_level,
                        'buffer': buffer}}}       
        return pars

    
    def from_hexrd_definitions(self, pars, det_num = 1):
        det_name = 'detector_{:d}'.format(det_num)
        stage_t  = pars['oscillation_stage']['translation']
        translation = pars['detectors'][det_name]['transform']['translation'] # in mm
        tilt        = pars['detectors'][det_name]['transform']['tilt'] # in radians
        pixels      = pars['detectors'][det_name]['pixels']
        
        O = [[None, None], [None, None]]
        if 'orientation' in pars['detectors'][det_name]['transform'].keys():
            hexrd_orientation = pars['detectors'][det_name]['transform']['orientation']
            if   hexrd_orientation == 'none': O = [[-1, 0], [ 0,-1]]
            elif hexrd_orientation == 't'   : O = [[ 0,-1], [-1, 0]]
            elif hexrd_orientation == 'v'   : O = [[ 1, 0], [ 0,-1]]
            elif hexrd_orientation == 'h'   : O = [[-1, 0], [ 0, 1]]
            elif hexrd_orientation == 'r180': O = [[ 1, 0], [ 0, 1]]
            elif hexrd_orientation == 'r90' : O = [[ 0, 1], [-1, 0]]
            elif hexrd_orientation == 'r270': O = [[ 0,-1], [ 1, 0]]
            
        while None in np.asarray(O):
            print('Image orientation (O-matrix) is missing!')
            x = input('Set it as 4 spaced numbers (O11 O12 O21 O22):').split()
            O = [[int(v) for v in x[0:2]], [int(v) for v in x[2:]]]
                                     
        det_pixsize_sf = np.asarray([pixels['size'][0], pixels['size'][1]]) # in um
        det_imgsize_sf = np.asarray([pixels['rows']   , pixels['columns']])

        fable_pixsize_zy = np.matmul(np.linalg.inv(O), det_pixsize_sf) # rot. about img. center
        fable_imgsize_zy = np.matmul(np.linalg.inv(O), det_imgsize_sf) # rot. about img. center
        fable_pixsize_sf =         abs(fable_pixsize_zy)
        fable_imgsize_sf = np.rint(abs(fable_imgsize_zy))

        ### HEXRD[x,y,z] = FABLE[-y,z,-x] ###
        r0 = Rotation.from_euler('y', -tilt[0])  # 1st is about HEXRD:+x (radians) = about FABLE:-y
        r1 = Rotation.from_euler('z',  tilt[1])  # 2nd is about HEXRD:+y (radians) = about FABLE:+z
        r2 = Rotation.from_euler('x', -tilt[2])  # 3rd is about HEXRD:+z (radians) = about FABLE:-x
        fable_tilt = (r2*r1*r0).as_euler('zyx')[::-1] # radians

        r0 = Rotation.from_euler('x', tilt[0])  # 1st is about HEXRD:+x (radians)
        r1 = Rotation.from_euler('y', tilt[1])  # 2nd is about HEXRD:+y (radians)
        r2 = Rotation.from_euler('z', tilt[2])  # 3rd is about HEXRD:+z (radians) 

        det_norm = (r2*r1*r0).apply( np.asarray([0,0,1]) ) # normal to the detector plane after tilts
        z_shift = (det_norm[0]*translation[0] + det_norm[1]*translation[1])/det_norm[2]
        fable_distance = - (translation[2] + z_shift) # FABLE +x is HEXRD -z

        hexrd_centpos_befor_rot = np.asarray([translation[0], translation[1], -z_shift])
        hexrd_centpos_after_rot = np.matmul( np.linalg.inv((r2*r1*r0).as_matrix()), hexrd_centpos_befor_rot)
        fable_beampos_zy = np.asarray([-hexrd_centpos_after_rot[1], hexrd_centpos_after_rot[0]])
        fable_beampos_sf = np.matmul(np.linalg.inv(O), fable_beampos_zy)/fable_pixsize_sf # rot. about img. center
        fable_centpos_sf = fable_beampos_sf + (fable_imgsize_sf-1)/2
        fable_wavelength = 12.39842/pars['beam']['energy'] # in keV
        fable_wedge      = 90-pars['beam']['vector']['polar_angle'] # in degrees, must be checked before using!
        fable_chi        = pars['oscillation_stage']['chi'] # in degrees, must be checked before using!
        buffer           = pars['detectors'][det_name]['buffer']
        saturation_level = pars['detectors'][det_name]['saturation_level']

        self.set_attr('wavelength', fable_wavelength)
        self.set_attr('wedge'     , np.radians(fable_wedge))
        self.set_attr('chi'       , np.radians(fable_chi))
        self.set_attr('buffer'    , buffer)
        self.set_attr('saturation_level', saturation_level)                                           
        self.set_attr('O'        , O)
        self.set_attr('tilt'     , fable_tilt)
        self.set_attr('distance' , fable_distance*1000) # im um
        self.set_attr('y_size'   , fable_pixsize_sf[1]*1000) # im um
        self.set_attr('z_size'   , fable_pixsize_sf[0]*1000) # im um
        self.set_attr('dety_size', int(fable_imgsize_sf[1]))
        self.set_attr('detz_size', int(fable_imgsize_sf[0]))
        self.set_attr('y_center' , fable_centpos_sf[1])
        self.set_attr('z_center' , fable_centpos_sf[0])
        
        return self

    
    def move_detector(self, shift_xyz_in_mm): # in FABLE coordinates
        self.add_to_log(f'Moving detector by: {shift_xyz_in_mm} (XYZ in mm)', False)
        pars = self.into_hexrd_definitions()
        det_name = 'detector_{:d}'.format(1)
        hexrd_translation = np.asarray( pars['detectors'][det_name]['transform']['translation'] ) # in mm 
        addit_translation = np.asarray([-shift_xyz_in_mm[1], shift_xyz_in_mm[2], -shift_xyz_in_mm[0]]) # in HEXRD coordinates
        pars['detectors'][det_name]['transform']['translation'] = list(hexrd_translation + addit_translation)
        self.from_hexrd_definitions(pars = pars)
        return
    
    
    def load_yml(self, directory = None, yml_file = None, det_num = 1):
        if directory: self.set_attr('directory', directory)
        if yml_file : self.set_attr('name', yml_file.replace('.yml', ''))
        self.add_to_log(f'Reading file: {self.directory+self.name}.yml', False)
        if not os.path.isfile(self.directory+self.name+'.yml'): raise FileNotFoundError
        with open(self.directory+self.name+'.yml', "r") as f:
            pars = yaml.safe_load(f)
        self.from_hexrd_definitions(pars = pars, det_num = det_num)            
        self.add_to_log('File closed!', False)
        return
    
    
    def save_yml(self, directory = None, yml_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if yml_file : self.set_attr('name', yml_file.replace('.yml', ''))
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        self.add_to_log(f'Writing file: {self.directory+self.name}.yml', False)
        
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.name+'.yml'):
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
                    self.set_attr('name', x.replace('.yml',''))

        pars = self.into_hexrd_definitions(det_num = 1)
        with open(self.directory+self.name+'.yml', "w") as f:
            yaml.dump(pars, f)
        self.add_to_log('File closed!', False)
        return                  
  

    def load_poni(self, directory = None, poni_file = None):
        if directory: self.set_attr('directory', directory)
        if poni_file: self.set_attr('name', poni_file.replace('.poni',''))
        self.add_to_log(f'Reading file: {self.directory+self.name}.poni', False)
        if not os.path.isfile(self.directory+self.name+'.poni'): raise FileNotFoundError

        with open(self.directory+self.name+'.poni', "r") as f:
            for line in f:
                if len(line) < 2: continue
                if line[0] == '#' and 'orientation' in line:
                    m = line.split('[[')[1].split(']]')[0]
                    O = [int(v) for v in m.replace('], [',' ').replace(', ',' ').split()]
                    self.set_attr('O', [[O[0], O[1]], [O[2], O[3]]])
                words = line.split() # words in line
                if words[1] in ['None','none', 'nan']: continue
                if('Detector_config' in words[0]):
                    pixel1 = 1e6*float(words[2].replace(',', '')) # in m
                    pixel2 = 1e6*float(words[4].replace(',', '')) # in m
                    max_shape1 = int(words[6].replace('[', '').replace(',', ''))
                    max_shape2 = int(words[7].replace(']', '').replace('}', ''))
                if('Distance' in words[0]): distance = 1e6*float(words[1])
                if('Poni1' in words[0]): poni1 = 1e6*float(words[1]) # in m
                if('Poni2' in words[0]): poni2 = 1e6*float(words[1]) # in m
                if('Rot1' in words[0]): rot1 = float(words[1])
                if('Rot2' in words[0]): rot2 = float(words[1])
                if('Rot3' in words[0]): rot3 = float(words[1])
                if('Wavelength' in words[0]): wavelength = 1e10*float(words[1]) # in m
        f.close()
        self.add_to_log('File closed!', False)
        
        while None in np.asarray(self.O):
            print('Image orientation (O-matrix) is missing!')
            x = input('Set it as 4 spaced numbers (O11 O12 O21 O22):').split()
            self.set_attr('O', [[int(v) for v in x[0:2]], [int(v) for v in x[2:]]])
        
        self.set_attr('wavelength', wavelength)
        self.set_attr('dety_size', max_shape2) # (slow, fast)
        self.set_attr('detz_size', max_shape1) # (slow, fast)
        self.set_attr('y_size', pixel2)
        self.set_attr('z_size', pixel1)
        fable_tiltrot_zy = np.matmul(np.asarray(self.O), np.asarray([rot1, rot2])) # rot. about img. center
        self.set_attr('tilt', [rot3, fable_tiltrot_zy[1], fable_tiltrot_zy[0]])
        self.set_attr('distance', distance/np.cos(rot1)/np.cos(rot2))
        self.set_attr('y_center', -0.5 + (poni2-distance*np.tan(rot1))/pixel2)
        self.set_attr('z_center', -0.5 + (poni1+distance*np.tan(rot2)/np.cos(rot1))/pixel1)
        return

    
    def save_poni(self, directory = None, poni_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if poni_file: self.set_attr('name', poni_file.replace('.poni',''))
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        self.add_to_log(f'Writing file: {self.directory+self.name}.poni', False)

        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.name+'.poni'):
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
                    self.set_attr('name', x.replace('.poni', ''))

        wavelength = 1e-10*self.wavelength
        rot1,rot2 = np.matmul(np.linalg.inv(self.O), np.asarray([self.tilt[2], self.tilt[1]])) # rot. about img. center
        rot3 = self.tilt[0]
        pixel1 = 1e-6*self.z_size
        pixel2 = 1e-6*self.y_size
        max_shape = [self.detz_size, self.dety_size] # (slow, fast)
        distance = 1e-6*self.distance*np.cos(rot1)*np.cos(rot2)
        poni1 = (self.z_center+0.5)*pixel1 - distance*np.tan(rot2)/np.cos(rot1)
        poni2 = (self.y_center+0.5)*pixel2 + distance*np.tan(rot1)
        
        pars = self.into_hexrd_definitions()
        hexrd_orientation = pars['detectors']['detector_1']['transform']['orientation']

        f = open(self.directory+self.name+'.poni',"w")
        f.write('# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis \n')
        f.write(f'# Converted from FABLE using image orientation: {str(self.O)} = {hexrd_orientation} in HEXRD\n')
        f.write('poni_version: 2\nDetector: Detector\n')

        Detector_config = {"pixel1": pixel1, "pixel2": pixel2, "max_shape": max_shape}  # (slow, fast)
        f.write('Detector_config: ' + str(Detector_config).replace('\'', '\"') + '\n')
        f.write(f'Distance: {distance}\n')
        f.write(f'Poni1: {poni1}\n')
        f.write(f'Poni2: {poni2}\n')
        f.write(f'Rot1: {rot1}\n')
        f.write(f'Rot2: {rot2}\n')
        f.write(f'Rot3: {rot3}\n')
        f.write(f'Wavelength: {wavelength}\n')
        f.close()
        self.add_to_log('File closed!', False)
        return self    
    
    
    def average_geometries(list_of_geometries):
        from numpy import nanmean
        L = list_of_geometries
        G = L[0]
        n = len(L)
        a = nanmean([g.wavelength for g in L if g.wavelength!=None])
        G.wavelength = None if str(a)=='nan' else a
        
        a = nanmean([g.material['unitcell'][0] for g in L if g.material['unitcell'][0]!=None])
        G.material['unitcell'][0] = None if str(a)=='nan' else a
        
        a = nanmean([g.material['unitcell'][1] for g in L if g.material['unitcell'][1]!=None])
        G.material['unitcell'][1] = None if str(a)=='nan' else a
        
        a = nanmean([g.material['unitcell'][2] for g in L if g.material['unitcell'][2]!=None])
        G.material['unitcell'][2] = None if str(a)=='nan' else a
        
        a = nanmean([g.distance for g in L if g.distance!=None])
        G.distance = None if str(a)=='nan' else a

        a = nanmean([g.tilt[0] for g in L if g.tilt[0]!=None])
        G.tilt[0] = None if str(a)=='nan' else a

        a = nanmean([g.tilt[1] for g in L if g.tilt[1]!=None])
        G.tilt[1] = None if str(a)=='nan' else a

        a = nanmean([g.tilt[2] for g in L if g.tilt[2]!=None])
        G.tilt[2] = None if str(a)=='nan' else a

        a = nanmean([g.t[0] for g in L if g.t[0]!=None])
        G.t[0] = None if str(a)=='nan' else a

        a = nanmean([g.t[1] for g in L if g.t[1]!=None])
        G.t[1] = None if str(a)=='nan' else a

        a = nanmean([g.t[2] for g in L if g.t[2]!=None])
        G.t[2] = None if str(a)=='nan' else a
        
        a = nanmean([g.y_center for g in L if g.y_center!=None])
        G.y_center = None if str(a)=='nan' else a

        a = nanmean([g.z_center for g in L if g.z_center!=None])
        G.z_center = None if str(a)=='nan' else a
        
        for g in L:
    #         if g.directory != G.directory: raise ValueError('Directories are not consistent!')
            if g.material['unitcell'][3:] != G.material['unitcell'][3:]: raise ValueError('Unit cell anlges are not consistent!')
            if g.material['symmetry']     != G.material['symmetry']:     raise ValueError('Symmetries are not consistent!')
            if g.material['spacegroup']   != G.material['spacegroup']:   raise ValueError('Spacegroups are not consistent!')
    #         if g.chi != G.chi: raise ValueError('chis are not consistent!')
            if g.fit_tolerance != G.fit_tolerance: raise ValueError('fit_tolerances are not consistent!')
            if g.min_bin_prob != G.min_bin_prob: raise ValueError('min_bin_probs are not consistent!')
            if g.no_bins != G.no_bins: raise ValueError('no_bins are not consistent!')
            if g.O != G.O: raise ValueError('O-matrices are not consistent!')
            if g.omegasign != G.omegasign: raise ValueError('omegasigns are not consistent!')
    #         if g.wedge != G.wedge: raise ValueError('wedges are not consistent!')
            if g.weight_hist_intensities != G.weight_hist_intensities: raise ValueError('weight_hist_intensities are not consistent!')
            if g.y_size != G.y_size: raise ValueError('y_sizes are not consistent!')
            if g.z_size != G.z_size: raise ValueError('z_sizes are not consistent!')
            if g.dety_size != G.dety_size: raise ValueError('dety_sizes are not consistent!')
            if g.detz_size != G.detz_size: raise ValueError('detz_sizes are not consistent!')
        return G
    
    
    def save_geometries_as_yml(list_of_geometries, directory, yml_file, overwrite = False, overwrite_print=False):
        G = list_of_geometries[0]
        pars = G.into_hexrd_definitions(det_num = 1)
        for det_num, gm in enumerate(list_of_geometries):
            if gm.wavelength != G.wavelength: print('Warning! Wavelengths are not consistent!')
            if gm.omegasign  != G.omegasign:  print('Warning! Omegasigns are not consistent!')        
            if gm.wedge      != G.wedge:      print('Warning! Wedges are not consistent!')
            if gm.chi        != G.chi:        print('Warning! Chis are not consistent!')
            if gm.O          != G.O  :        print('Warning! O-matrices are not consistent!')
            if gm.t          != G.t  :        print('Warning! Stage translations are not consistent!')

            p = gm.into_hexrd_definitions(det_num = 1)
            if p['beam'] != pars['beam']: raise ValueError('Beam parameters are not consistent!')
            if p['id']   != pars['id']  : raise ValueError('Oscillation stages are not consistent!')
            pars['detectors'][f'detector_{det_num+1}'] = p['detectors'][f'detector_{1}']

        # Check if file exists, if yes then ask for the permission to overwrite    
        while os.path.isfile(directory+yml_file):
            if overwrite_print:
                print('Warning! File already exist!')
            if overwrite:
                if overwrite_print:
                    print('Overwriting...')
                break
            else:
                x = input('Type new name or ! to overwrite, a - abort:')
                if x in ['!']:
                    if overwrite_print:
                        print('Overwriting...')
                    break
                elif x in ['a', 'A']:
                    if overwrite_print:
                        print('Aborted!')
                    return
                else:
                    yml_file =  x

        with open(directory+yml_file, "w") as f:
            yaml.dump(pars, f)

        return directory+yml_file
    
    
    def ds_from_tth(self, tth):
        return 2*np.sin(np.radians(tth/2))/self.wavelength
    
    
    def tth_from_ds(self, ds):
        return 2*np.degrees(np.arcsin(ds*self.wavelength/2))