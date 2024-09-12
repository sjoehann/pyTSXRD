# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:00:49 2022

@author: sjö

Class to run PolyXSim.
"""

import os, subprocess
import numpy as np
from datetime import datetime
import pyTSXRD
from pyTSXRD.Grain import Grain
single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"
        
class PolySim:
    
    def __init__(self, directory = None):
        self.log = []
        self.directory = None
        self.inp_file = None
        self.geometry = pyTSXRD.Geometry()
        self.grains = []
        self.beamflux = None
        self.omega_start = None
        self.omega_step = None
        self.omega_end = None
        self.beampol_factor = None
        self.beampol_direct = None
        self.theta_min = None
        self.theta_max = None
        self.no_grains = None
        self.gen_U = None
        self.gen_pos = [None, None]
        self.gen_eps = [None, None, None, None, None]
        self.sample_xyz = [None, None, None]
        self.sample_cyl = [None, None]
        self.gen_size = [None, None, None, None]
        self.direc = None
        self.stem = None
        self.make_image = None
        self.output = []
        self.bg = None
        self.noise = 0
        self.psf = None
        self.peakshape = [None, None, None]
        self.add_to_log('Created PolySim object.', False)
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
        if attr == 'geometry' and old is not None: old = old.__dict__
        if attr == 'geometry' and new is not None: new = new.__dict__
        self.add_to_log(attr+': '+str(old)+' -> '+str(new))
        return
        
        
    def add_to_attr(self, attr, value):
        old_list = getattr(self, attr)
        if type(old_list) == list: 
            setattr(self, attr, old_list+[value])
            new_list = getattr(self, attr)
            self.add_to_log(attr+': += '+str(new_list[-1]))
        else:
            raise AttributeError('This attribute is not a list!')
        return
    
    
    def print(self, also_log = False):
        print(double_separator+'PolySim object:')
        print('directory:', self.directory)
        print('inp_file:', self.inp_file)
        print('beamflux:'   , self.beamflux)
        print('omega_start:', self.omega_start)
        print('omega_step:' , self.omega_step)
        print('omega_end:'  , self.omega_end)
        print('theta_min:'  , self.theta_min)
        print('theta_max:'  , self.theta_max)
        print('no_grains:'  , self.no_grains)
        print('gen_U:'      , self.gen_U)
        print('gen_pos:'    , self.gen_pos)
        print('gen_eps:'    , self.gen_eps)
        print('gen_size:'   , self.gen_size)
        print('sample_xyz:' , self.sample_xyz)
        print('sample_cyl:' , self.sample_cyl)
        print('direc:'      , self.direc)
        print('stem:'       , self.stem)
        print('make_image:' , self.make_image)
        print('output:'     , self.output)
        print('bg:'         , self.bg)
        print('noise:'      , self.noise)
        print('psf:'        , self.psf)
        print('peakshape:'  , self.peakshape)
        if self.geometry: self.geometry.print()   
        for i,g in enumerate(self.grains):
            print(  f'grain {i}:')
            g.print()
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return
    

    def load_inp(self, directory = None, inp_file = None):
        if directory: self.set_attr('directory', directory)
        if inp_file: self.set_attr('inp_file', inp_file)
        self.add_to_log(f'Reading file: {self.directory+self.inp_file}', False)
        if not os.path.isfile(self.directory+self.inp_file): raise FileNotFoundError
        
        GM = pyTSXRD.Geometry(directory = directory)
        O = [[None, None], [None, None]]
        grs = []
        tilt = [None, None, None]
        material = dict(self.material)
        with open(self.directory+self.inp_file, "r") as f:
            for line in f:
                if line[0] == '#' or len(line) < 2: continue
                words = line.split() # words in line
                if words[1] in ['None','none', 'nan']: continue
                if ('beamflux' in words[0]):
                    self.set_attr('beamflux', float(words[1]))
                elif ('wavelength' in words[0]):
                    GM.set_attr('wavelength', float(words[1]))
                elif ('distance' in words[0]):
                    GM.set_attr('distance', 1000*float(words[1]))
                elif ('dety_center' in words[0]):
                    y_center = float(words[1])
                elif ('detz_center' in words[0]):
                    z_center = float(words[1])
                elif ('dety_size' in words[0]):
                    dety_size = int(words[1])
                elif ('detz_size' in words[0]):
                    detz_size = int(words[1])
                elif ('y_size' in words[0]):
                    y_size = 1000*float(words[1]) # in um
                elif ('z_size' in words[0]):
                    z_size = 1000*float(words[1]) # in um
                elif ('tilt_x' in words[0]):
                    tilt[0] = float(words[1])
                elif ('tilt_y' in words[0]):
                    tilt[1] = float(words[1])
                elif ('tilt_z' in words[0]):
                    tilt[2] = float(words[1])
                elif ('omega_start' in words[0]):
                    self.set_attr('omega_start', float(words[1]))
                elif ('omega_end' in words[0]):
                    self.set_attr('omega_end', float(words[1]))
                elif ('omega_step' in words[0]):
                    self.set_attr('omega_step', float(words[1]))
                elif ('omega_sign' in words[0]):
                    GM.set_attr('omegasign', int(float(words[1])))
                elif ('beampol_factor' in words[0]):
                    self.set_attr('beampol_factor', float(words[1]))
                elif ('beampol_direct' in words[0]):
                    self.set_attr('beampol_direct', float(words[1]))
                elif ('theta_min' in words[0]):
                    self.set_attr('theta_min', float(words[1]))
                elif ('theta_max' in words[0]):
                    self.set_attr('theta_max', float(words[1]))
                elif ('o11' in words[0]): O[0][0] = int(words[1])
                elif ('o12' in words[0]): O[0][1] = int(words[1])
                elif ('o21' in words[0]): O[1][0] = int(words[1])
                elif ('o22' in words[0]): O[1][1] = int(words[1])
                elif ('no_grains' in words[0]):
                    self.set_attr('no_grains', int(words[1]))
                elif ('gen_U' in words[0]):
                    self.set_attr('gen_U', int(words[1]))
                elif ('gen_pos' in words[0]):
                    self.set_attr('gen_pos', [int(words[1]), int(words[2])])
                elif ('gen_eps' in words[0]):
                    self.set_attr('gen_eps', [int(words[1])]+[float(v) for v in words[2:6]])
                elif ('gen_size' in words[0]):
                    self.set_attr('gen_size', [int(words[1])]+[float(v) for v in words[2:5]])
                elif ('sample_xyz' in words[0]):
                    self.set_attr('sample_xyz', [float(v) for v in words[1:4]])
                elif ('sample_cyl' in words[0]):
                    self.set_attr('sample_cyl', [float(v) for v in words[1:3]])
                elif ('U_grains_' in words[0]):
                    ind = int(words[0].replace('U_grains_','')) 
                    if ind > len(grs)-1:
                        grs.append(Grain( directory = directory, grain_id = ind))
                    grs[ind].set_attr('u', [float(v) for v in words[1:10]])
                elif ('pos_grains_' in words[0]):
                    ind = int(words[0].replace('pos_grains_','')) 
                    if ind > len(grs)-1:
                        grs.append(Grain( directory = directory, grain_id = ind))
                    grs[ind].set_attr('position', [float(v) for v in words[1:10]])
                elif ('eps_grains_' in words[0]):
                    ind = int(words[0].replace('eps_grains_','')) 
                    if ind > len(grs)-1:
                        grs.append(Grain( directory = directory, grain_id = ind))
                    grs[ind].set_attr('eps', [float(v) for v in words[1:7]])
                elif ('size_grains_' in words[0]):
                    ind = int(words[0].replace('size_grains_','')) 
                    if ind > len(grs)-1:
                        grs.append(pyTSXRD.Grain( directory = directory, grain_id = ind))
                    grs[ind].set_attr('size', float(words[1]))
                elif ('unit_cell' in words[0]):
                    material['unitcell'] = [float(v) for v in words[1:7]]
                elif ('sgno' in words[0]):
                    material['spacegroup'] = int(words[1])
                elif ('direc' in words[0]):
                    self.set_attr('direc', words[1].replace("'", '').replace('./', self.directory))
                elif ('stem' in words[0]):
                    self.set_attr('stem', words[1].replace("'", ''))
                elif ('make_image' in words[0]):
                    self.set_attr('make_image', int(words[1]))
                elif ('stem' in words[0]):
                    self.set_attr('stem', words[1].replace("'", ''))
                elif ('output' in words[0]):
                    x = line.find("#")
                    words2 = line[0:x].split()
                    self.set_attr('output', [v.replace("'", '') for v in words2[1::]] )
                elif ('bg' in words[0]):
                    self.set_attr('bg', float(words[1]))
                elif ('noise' in words[0]):
                    self.set_attr('noise', float(words[1]))
                elif ('psf' in words[0]):
                    self.set_attr('psf', float(words[1])) 
                elif ('peakshape' in words[0]):
                    self.set_attr('peakshape', [float(v) for v in words[1:4]])
        f.close()
        
        while None in np.asarray(O):
            print('Image orientation (O-matrix) is missing!')
            x = input('Set it as 4 spaced numbers (O11 O12 O21 O22):').split()
            O = [[int(v) for v in x[0:2]], [int(v) for v in x[2:]]]
            
        GM.set_attr('material', material)
        GM.set_attr('O', O)
        GM.set_attr('tilt', tilt)
        
        # Conversion from lab-related parameters (PolyXSim) to the detector-related ones (ImageD11), i.e. applying the inverse detector flips.
        fable_imgsize_zy = np.asarray([detz_size  , dety_size  ]) # in FABLE ZY coordinates
        fable_pixsize_zy = np.asarray([z_size     , y_size     ]) # in um here
        fable_beampos_zy = np.asarray([z_center   , y_center   ]) # in pixels
        fable_centpos_zy = (fable_imgsize_zy-1)/2

        det_imgsize_sf = abs(np.matmul(np.linalg.inv(O), fable_imgsize_zy)) # in pixels
        det_pixsize_sf = abs(np.matmul(np.linalg.inv(O), fable_pixsize_zy)) # in um
        det_beampos_sf =     np.matmul(np.linalg.inv(O), (fable_beampos_zy-fable_centpos_zy)*fable_pixsize_zy) # in um from img. center
        centpos_sf = det_beampos_sf/det_pixsize_sf + (det_imgsize_sf-1)/2 # convert to pixels from detector (0,0)

        GM.set_attr('y_center' , centpos_sf[1])
        GM.set_attr('z_center' , centpos_sf[0])
        GM.set_attr('y_size'   , det_pixsize_sf[1])
        GM.set_attr('z_size'   , det_pixsize_sf[0])
        GM.set_attr('dety_size', round(det_imgsize_sf[1]))
        GM.set_attr('detz_size', round(det_imgsize_sf[0]))

        for g in grs: g.set_attr('spacegroup', GM.spacegroup)
        self.set_attr('grains', grs)
        self.set_attr('geometry', GM)     
        self.add_to_log('File closed!', False)
        return
    
            
    def save_inp(self, directory = None, inp_file = None, overwrite = False):
        if directory: self.set_attr('directory', directory)
        if inp_file: self.set_attr('inp_file', inp_file)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        self.add_to_log(f'Writing file: {self.directory+self.inp_file}', False)
        
        # Check if file exists, if yes then ask for the permission to overwrite
        while os.path.isfile(self.directory+self.inp_file):
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
                    self.set_attr('inp_file', x)
        
        # Conversion from detector-related parameters (ImageD11) to the lab-related ones (PolyXSim), i.e. applying the detector flips.
        fable_distance = self.geometry.distance/1000 # in mm here
        det_imgsize_sf = np.asarray([self.geometry.detz_size, self.geometry.dety_size]) # (det_slow, det_fast)
        det_pixsize_sf = np.asarray([self.geometry.z_size   , self.geometry.y_size   ])/1000 # in mm here
        det_beampos_sf = np.asarray([self.geometry.z_center , self.geometry.y_center ]) # in pixels
        det_centpos_sf = (det_imgsize_sf-1)/2

        fable_beampos_zy = np.matmul(np.asarray(self.geometry.O), (det_beampos_sf-det_centpos_sf)*det_pixsize_sf) # rot. about img. center
        fable_pixsize_zy = abs(np.matmul(np.asarray(self.geometry.O), det_pixsize_sf)) # rot. about img. center
        fable_imgsize_zy = abs(np.matmul(np.asarray(self.geometry.O), det_imgsize_sf)) # rot. about img. center
        centpos_zy = fable_beampos_zy/fable_pixsize_zy + (fable_imgsize_zy-1)/2
    
        f = open(self.directory+self.inp_file ,"w")
        f.write('### Instrumental\n')
        f.write('wavelength  {}   # in angstrom\n'.format(self.geometry.wavelength))
        if self.beamflux:
            f.write('beamflux  {:.2e}   # beam flux (photons/sec/mm^2)\n'.format(self.beamflux))
        else:
            f.write('beamflux  {}   # beam flux (photons/sec/mm^2)\n'.format(self.beamflux))
        
        f.write('distance  {}   # sample-detector distance (mm)\n'.format(fable_distance))
        f.write('dety_center  {} # beamcenter, y in pixel coordinatees\n'.format(centpos_zy[1]))
        f.write('detz_center  {} # beamcenter, z in pixel coordinatees\n'.format(centpos_zy[0]))
        f.write('y_size  {}         # Pixel size y (mm)\n'.format(fable_pixsize_zy[1]))
        f.write('z_size  {}         # Pixel size z (mm)\n'.format(fable_pixsize_zy[0]))
        f.write('dety_size  {}    # detector y size (pixels)\n'.format(round(fable_imgsize_zy[1])))
        f.write('detz_size  {}    # detector z size (pixels)\n'.format(round(fable_imgsize_zy[0])))
        
        f.write('tilt_x  {}         '.format(self.geometry.tilt[0]))
        f.write('# detector tilt counterclockwise around lab x axis (beam direction) in rad\n')
        f.write('tilt_y  {}         '.format(self.geometry.tilt[1]))
        f.write('# detector tilt counterclockwise around lab y axis in rad\n')
        f.write('tilt_z  {}         '.format(self.geometry.tilt[2]))
        f.write('# detector tilt counterclockwise around lab z axis (same as omega) in rad\n')
        
        f.write('omega_start  {}    # Minimum Omega in range of interest (in deg)\n'.format(self.omega_start))
        f.write('omega_end  {}   # Maximum Omega in range of interest (in deg)\n'.format(self.omega_end))
        f.write('omega_step  {}      # Omega step size (in deg)\n'.format(self.omega_step))
        f.write('omega_sign  {}       # Sign of omega rotation\n'.format(self.geometry.omegasign))
        f.write('beampol_factor  {}    # Polarisation factor\n'.format(self.beampol_factor))
        f.write('beampol_direct  {}    # Polarisation direction\n'.format(self.beampol_direct))
        f.write('theta_min  {}         # Minimum theta angle for reflection generation\n'.format(self.theta_min))
        f.write('theta_max  {}         # Maximum theta angle for reflection generation\n'.format(self.theta_max))
        f.write('o11  {}              # Orientation matrix of detector\n'.format(self.geometry.O[0][0]))
        f.write('o12  {}              # [[o11,o12]\n'.format(self.geometry.O[0][1]))
        f.write('o21  {}              # [o21,o22]]\n'.format(self.geometry.O[1][0]))
        f.write('o22  {}              #\n'.format(self.geometry.O[1][1]))

        f.write('\n### Grains++\n')
        f.write('no_grains {}         # number of grains\n'.format(self.no_grains))
        f.write('gen_U {}              # generate grain orientations\n'.format(self.gen_U))
        f.write('gen_pos {} {}         # generate grain positions\n'.format(self.gen_pos[0], self.gen_pos[1]))
        f.write('gen_eps {} {} {} {} {}'.format(self.gen_eps[0], self.gen_eps[1], self.gen_eps[2], self.gen_eps[3], self.gen_eps[4]))
        f.write('    # generate strain tensors [mean (diag) spread (diag) mean (off-diag) spread (off-diag)]\n')
        if not None in self.sample_xyz:
            f.write('sample_xyz {} {} {}'.format(self.sample_xyz[0], self.sample_xyz[1], self.sample_xyz[2]))
            f.write('     # sample size in mm (only one of sample_xyz and sample_cyl can be given)\n')                
        
        f.write('sample_cyl {} {}'.format(self.sample_cyl[0], self.sample_cyl[1]))
        f.write('         # cylinder dimension, diameter and length, of sample in mm\n')
        
        f.write('gen_size {} {} {} {}'.format(self.gen_size[0], self.gen_size[1], self.gen_size[2], self.gen_size[3]))
        f.write('   # generate grain sizes (0=off, 1=on) [mean minimum maximum]\n')
        f.write('                            # grain sizes for log normal distribution in mm\n')
        
        if self.gen_U == 0:
            for n,g in enumerate(self.grains):
                s = [f'U_grains_{n}']+[f'{v}' for v in np.asarray(g.u).flatten()]
                f.write(' '.join(s) + '\n')
        
        if self.gen_pos[0] == 0:
            for n,g in enumerate(self.grains):
                s = [f'pos_grains_{n}']+[f'{v}' for v in g.position]
                f.write(' '.join(s) + '\n' )
            
        if self.gen_eps[0] == 0:
            for n,g in enumerate(self.grains):
                s = [f'eps_grains_{n}']+[f'{v}' for v in np.asarray(g.eps).flatten()]
                f.write(' '.join(s) + '\n' )

        if self.gen_size[0] == 0:
            for n,g in enumerate(self.grains):
                f.write(f'size_grains_{n} {g.size}\n')
    
        f.write('\n### Structural\n')
        try:
            s =[f'{v:.6f}' for v in self.geometry.material['unitcell']]
        except:
            s = ['None']*6
        f.write('unit_cell '+' '.join(s) + '\n' )
        f.write('sgno  {}               # space group number\n'.format(self.geometry.material['spacegroup']))
        
        f.write('\n### Files\n')
        f.write('direc  \'{}\'            # working directory\n'.format(self.direc))
        f.write('stem  \'{}\'     # prefix of all generated files\n'.format(self.stem))
        
        f.write('\n### Images\n')
        f.write('make_image {}          # images produced unless 0 is given\n'.format(self.make_image))
        if self.output: s = [f'\'{v}\'' for v in self.output]
        else: s = ['None']
        f.write('output ' + ' '.join(s) + '   # possible: \'.edf\' and \'.tif\' for image types\n')
        f.write('bg {}                 # background intensity counts\n'.format(self.bg))
        f.write('noise {}               # add random poisson noise\n'.format(self.noise))
        f.write('psf {}               # apply detector point spread function, 0=no psf\n'.format(self.psf))
        f.write('                      # The input value corresponds roughly to the spread in pixels in every direction\n')
        f.write('peakshape {} {} {}'.format(self.peakshape[0], self.peakshape[1], self.peakshape[2]))
        f.write('     # type, spread (in pixels from the peak centre), (spread in omega for type 1)\n')
        f.write('                      # type=0 spike, type=1 3D Gaussian')          
        f.close()
        self.add_to_log('File closed!', False)
        return
    
    
    def run_PolyXsim(self, directory = None, inp_file = None):
        if directory: self.set_attr('directory', directory)
        if inp_file:
            self.set_attr('inp_file', inp_file)
            self.load_inp()

        del_old  = subprocess.call('rm '+self.direc+self.stem+'*', shell=True)
        if del_old > 0: self.add_to_log(f"Deleted old files.", False)           
        self.save_inp(overwrite = True)

        command = 'PolyXSim.py -i ' + self.directory+self.inp_file
        self.add_to_log(f'Running: '+command, False)
        process = subprocess.run(command.split(), check=True,stdout=subprocess.PIPE, universal_newlines=True)
        self.add_to_log('Output:'+process.stdout, False)
        #print('Last line in the output:'+process.stdout.splitlines()[-1])
        if 'show this help message and exit' in process.stdout.splitlines()[-1]:
            raise ChildProcessError('PolyXsim failed to run!')
        
        # Corection for geometry bug in PolyXSim
        subprocess.call('cp '+self.direc+self.stem+'.par '+self.direc+self.stem+'_PolyXSim.par ', shell=True)
        GM = pyTSXRD.Geometry()
        GM.material = dict(self.geometry.material)
        if self.direc[0:2] == './':
            GM.load_par(directory = self.directory + self.direc[2:], par_file = self.stem + '.par')
        else:
            GM.load_par(directory = self.direc, par_file = self.stem + '.par')
        #self.geometry.print()
        self.geometry.set_attr('material', GM.material)
        self.geometry.set_attr('chi', GM.chi)
        self.geometry.set_attr('fit_tolerance', GM.fit_tolerance)
        self.geometry.set_attr('wedge', GM.wedge)
        self.geometry.save_par(directory = GM.directory, par_file = self.stem + '.par', overwrite = True)
        return