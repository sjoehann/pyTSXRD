# -*- coding: utf-8 -*-
"""
Created on Wed Feb 02 11:00:00 2022

@author: sjö

Class to work with a single sweep scan. Loads, process, and peaksearches image files.
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
from ImageD11 import columnfile, blobcorrector
import pyTSXRD
from pyTSXRD.angles_and_ranges import merge_overlaps

single_separator = "--------------------------------------------------------------\n"
double_separator = "==============================================================\n"


class SweepProcessor:
    """Class to work with a single sweep scan. Loads, process, and peaksearches image files."""
    
    def __init__(self, directory=None, name=None):
        self.log = [] # log entries. Useful when you don't remember how exactly the data was analyzed.
        self.directory = None # Output directory.
        self.name = None # Name of an object and also the prefics for output files.
        self.position = [] # Sample stage position.
        self.log_meta = None # Information about the data.
        self.fio_meta = None # Information loaded from *fio file.
        self.sweep = {'fio_path':None, 'omega_start':None, 'omega_step':None,
                      'directory':None, 'stem':None, 'ndigits':None, 'ext':None} # Parameters of the sweep scan.
        self.chunk = {'frames':[], 'filenames':[], 'omegas':[]} # This describes the selection.
        self.imgs = None # Images.
        self.processing = {'options':None, 'bckg_indices':None, 'bckg':None} # Parameters of image processing.
        self.projs = {'imsum':None, 'immax':None, 'q0_pos':None,
                      'etatth':None, 'omgtth':None, 'omgeta':None} # Projections.
        self.peaksearch_thrs = [] # Thresholds for peaksearch
        self.peakmerge_thrs = [] # Thresholds for merging results from peaksearch.
        self.pix_tol    = None # Minimal distance between the peaks.
        self.geometry   = pyTSXRD.Geometry() # Geometry parameters for this dataset.
        self.add_to_log('Initialized SweepProcessor object.', False)
        if directory: self.set_attr('directory', directory)
        if name     : self.set_attr('name'     , name)
        return
 

    def add_to_log(self, str_to_add, also_print = False):
        """Method to add an entry to the log base of the object. This entry can be optionally printed."""
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
        if attr == 'geometry' and old is not None: old = old.__dict__ # This is because geometry is a class object.
        if attr == 'geometry' and new is not None: new = new.__dict__
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
        """Method to print the values of an object. The log is not printed by default."""
        print(double_separator+'SweepProcessor object:')
        print('directory:' , self.directory)
        print('name:'      , self.name)
        print('position:'  , self.position)
        
        if self.log_meta:
            print('log_meta:', list(self.log_meta['entries'][0].keys()))
            print('total:', len(self.log_meta['entries']), 'entries with keys:')
            print([k for k in self.log_meta['entries'][0].keys()] )
            
        if self.fio_meta:
            print('fio_meta:', list(self.fio_meta['entries'][0].keys()))
            print('total:', len(self.fio_meta['entries']), 'entries with keys:')
            print([k for k in self.fio_meta['entries'][0].keys()] )
    
    
        print('sweep:'     , self.sweep, '\n')
        print('chunk:'     , self.chunk, '\n')
        
        try: print('imgs:'  , len(self.imgs), 'of images', '\n')
        except: print('imgs:', self.imgs, '\n')
        
        print('processing:' , self.processing, '\n')
        print('projs:'      , self.projs, '\n')
        print('peaksearch_thrs:', self.peaksearch_thrs, '\n')
        print('peakmerge_thrs:', self.peakmerge_thrs, '\n')
        print('pix_tol:'    , self.pix_tol, '\n')
        if self.geometry: self.geometry.print()        
        if also_log:
            print(single_separator + 'Log:')
            for record in self.log: print(record)
        return
    
    
    def load_sweep(self, omega_start=None, omega_step=None,
                   directory=None, stem=None, ndigits=None, ext=None, frames = None):
        """Method to load the images from the specified directory which have a specified pattern in file names."""
        s = self.sweep
        c = self.chunk
        if omega_start is not None: s['omega_start'] = omega_start
        if omega_step  is not None: s['omega_step']  = omega_step
        if directory  : s['directory']   = directory
        if stem       : s['stem']        = stem
        if ndigits    : s['ndigits']     = ndigits
        if ext        : s['ext']         = ext
        if frames     : c['frames']      = frames
        
        #print(single_separator+'Loading sweep:', s['directory']+s['stem']+'*'+s['ext'], '...')
        all_files = sorted(os.listdir(s['directory']))
        regex = re.compile(r'\d+')
        check_stem = True
        check_digits = True
        for fname in all_files:
            stem, ext = os.path.splitext(fname)
            if ext in ['.tif', '.cbf', '.edf', '.hdf5', '.h5','nx5']:
                if ext != s['ext']:
                    #print('Detected file extension \''+ext+'\' vs provided \''+str(s['ext'])+'\'.')
                    x = input('Type: ! to overwrite, k - keep:')
                    if x in ['!']: s['ext'] = ext
                dig_part  = [x for x in regex.findall(stem)][-1]
                dig_len   = len(dig_part)
                stem_part = stem.replace(dig_part, '')
                if check_stem and stem_part != s['stem']:
                    #print('Detected file stem \''+stem_part+'\' vs provided \''+str(s['stem'])+'\'.')
                    x = input('Type: ! to overwrite, k - keep, d - keep and don\'t ask anymore:')
                    if x in ['!']: s['stem'] = stem_part
                    if x in ['d']: check_stem = False
                if check_digits and dig_len != s['ndigits']:
                    #print(f'Detected {dig_len} digits in file name vs provided '+str(s['ndigits']))
                    if s['ndigits'] == 'auto':
                        s['ndigits'] = dig_len
                    else:                    
                        x = input('Type: ! to overwrite, k - keep, d - keep and don\'t ask anymore:')
                        if x in ['!']: s['ndigits'] = dig_len
                        if x in ['d']: check_digits = False
        
#         if s['ext'] in ['.cbf', '.tif'] and 'p21.2' in self.directory:
#             import cbftiffmxrdfix
        #print(all_files)
        matching_files = [f for f in all_files if (s['stem'] in f and s['ext'] in f)]
        n0 = int(matching_files[0].replace(s['stem'],'').replace(s['ext'],''))
        for i in range(c['frames'][-1]):
            expected_file = s['stem']+str(n0+i).zfill(s['ndigits'])+s['ext']
            if expected_file not in matching_files:
                raise FileNotFoundError(expected_file)
                                  
        try:
            c['filenames'] = [matching_files[ii] for ii in c['frames']]
        except:
            raise FileNotFoundError('Some files are missing in the raw directory! Not uploaded yet?')
            
        c['omegas'] = np.empty([len(c['frames']),2], dtype=float)
        for ii in range(len(c['frames'])):
            start_angle = s['omega_start']+s['omega_step']*c['frames'][ii]
#             c['omegas'][ii] = [start_angle-0.5*s['omega_step'], start_angle+0.5*s['omega_step']]
            c['omegas'][ii] = [start_angle, start_angle + s['omega_step']]
        
        if s['omega_step'] > 0:
            imgs_omegas = c['omegas'][::1,::1]
            imgs_frames = c['frames'][::1]
        else: # CRUTCH for imageseries!
            imgs_omegas = c['omegas'][::-1,::-1]
            imgs_frames = c['frames'][::-1]
            
        if s != self.sweep: self.set_attr('sweep', s)
        if c != self.chunk: self.set_attr('chunk', c)
        self.add_to_log('Loading sweep: '+str(s), True)
        self.add_to_log('Selecting chunk: '+str(c), False)
        #np.save(s['directory']+self.name+"omega.npy", imgs_omegas)

        if s['ext'] in ['.tif', '.cbf', 'edf']: # Varex or Pilatus detector.
            config_file = self.directory+str(self.name)+"_sweepparam.yml"
            with open(config_file, 'w') as f:
                f.write('image-files:\n  directory: '+s['directory'])
                f.write('\n  files: "')
                #imgs = []
                for fname in c['filenames']:
                    f.write(fname+'\n')
                    #imgs.append(imageio.imread(s['directory']+fname))
                f.write('\"\noptions:  \n  empty frames: 0\n  max-frames: 0')
                f.write('\nmeta:\n  omega: ')#'\"! load-numpy-array omega.npy\"')
            f.close()
            imgs = imageseries.open(config_file, 'image-files')
            imgs.metadata['omega'] = imgs_omegas

        elif s['ext'] in ['.h5','.hdf5','nx5']: # Eiger detector
            config_file = self.directory+str(self.name)+"_sweepparam.yml"
            with open(config_file, 'w') as f:
                #f.write('image-files:\n  directory: '+s['directory'])
                f.write('hdf5:\n  directory: '+s['directory'])
                f.write('\n  files: "')
                #imgs = []
                for fname in c['filenames']:
                    f.write(fname+'\n')
                    #imgs.append(imageio.imread(s['directory']+fname))
                f.write('\"\noptions:  \n  empty frames: 0\n  max-frames: 0')
                f.write('\nmeta:\n  omega: ')#'\"! load-numpy-array omega.npy\"')
            f.close()
            imgs = imageseries.open(config_file, 'image-files')
            imgs.metadata['omega'] = imgs_omegas
    
        self.set_attr('imgs', imgs)
        del imgs_omegas, imgs_frames, all_files, matching_files, config_file
        self.add_to_log(f'{len(imgs)} images of size {imgs[0].shape} (slow, fast) loaded.', False)
        return
    
    
    def process_imgs(self, options=None, bckg=None):
        """Method to process the images after load_sweep. Corrects for background and detector flips."""
        p = self.processing
        if options is not None: p['options'] = options
        
        #print('Processing sweep with options:', str(p['options']), '...')
        if bckg is None:
            p['bckg_indices'] = []
            p['bckg'] = np.zeros( (self.imgs[0].shape), float)
        elif 'auto' in bckg: # 'auto80' can be used for example.
            try:
                Nmax = min( len(self.imgs), int(bckg.replace('auto', '')) )
                if Nmax<1 or Nmax>200: raise ValueError(bckg+' - Nmax must be >1 and < 200')
            except:
                Nmax = min( len(self.imgs), 50 )
            p['bckg_indices'] = list(range(1,len(self.imgs)-1,round(len(self.imgs)/Nmax) )) # ~Nmax images
            p['bckg'] = calculate_bckg(self.imgs, p['bckg_indices'])
            self.add_to_log(f"Calculated background using {len(p['bckg_indices'])} images", False)
        else:
            p['bckg_indices'] = []
            p['bckg'] = bckg
            
        try   : p['bckg'].shape == self.imgs[0].shape
        except: raise TypeError(type(p['bckg'])+' - incorrect background!')
        
        ProcessedIS = imageseries.process.ProcessedImageSeries
        if p['options']: imgs = ProcessedIS(self.imgs, [('dark', p['bckg']), p['options']])
        else           : imgs = ProcessedIS(self.imgs, [('dark', p['bckg'])])
        self.set_attr('processing', p)
        self.set_attr('imgs', imgs)
        self.geometry.set_attr('dety_size', imgs[0].shape[1]) # fast dimension in numpy array
        self.geometry.set_attr('detz_size', imgs[0].shape[0]) # slow dimension in numpy array
        omegastep = self.chunk['omegas'][0][1] - self.chunk['omegas'][0][0]
        self.geometry.set_attr('omegasign', int(np.sign(omegastep)))
        return

    def calculate_projections(self, q0_pos=None, rad_ranges = None):
        """Method to compute the omega-eta, omega-tth, and other projections. If needed, the images are masked for a certain radial range beforehand."""
        if not q0_pos or q0_pos == 'auto':
            q0_pos = [self.geometry.z_center, self.geometry.y_center]
        if type(q0_pos[1]) not in [int, float, np.float64]:
            raise ValueError('The y_center position is not defined properly!')
        if type(q0_pos[0]) not in [int, float, np.float64]:
            raise ValueError('The z_center position is not defined properly!')

        im_size = list(self.imgs[0].shape)
        min_r_0 = max(0 - q0_pos[0], q0_pos[0] - im_size[0], 0)
        min_r_1 = max(0 - q0_pos[1], q0_pos[1] - im_size[1], 0)
        max_r_0 = max(q0_pos[0] - 0, im_size[0] - q0_pos[0])
        max_r_1 = max(q0_pos[1] - 0, im_size[1] - q0_pos[1])
        min_rad = round(np.sqrt(min_r_0**2+min_r_1**2))
        max_rad = round(np.sqrt(max_r_0**2+max_r_1**2))

        if type(rad_ranges) != list: rad_ranges = [[min_rad, max_rad]]

        corrected_ranges = []
        for ir, rng in enumerate(rad_ranges):
            if rng[0] < min_rad: rng[0] = min_rad
            if rng[1] > max_rad: rng[1] = max_rad
            rng = [round(rng[0]), round(rng[1])]
            if rng[1] > rng[0] and rng not in corrected_ranges:
                corrected_ranges.append( rng )
            else:
                continue
        rad_ranges = merge_overlaps(corrected_ranges, margin=0)

        #print(f'Calculating projections for im_size={im_size}, q0_pos={q0_pos}, min_rad={min_rad}, max_rad={max_rad}\n...')
        immax = imageseries.stats.max(self.imgs, len(self.imgs))
        imsum = 1*self.imgs[0]
        imgp = warp_polar(1.*self.imgs[0], center=q0_pos, radius=max_rad)
        etatth = imgp
        omgeta = np.zeros([len(self.imgs), imgp.shape[0]], dtype=np.float32)
        omgtth = np.zeros([len(self.imgs), imgp.shape[1]], dtype=np.float32)
        omgeta[0,:] = np.nansum(imgp, 1)
        omgtth[0,:] = np.nansum(imgp, 0)

        etatth_rngs = []
        omgeta_rngs = []
        omgtth_rngs = []
        for ir, rng in enumerate(rad_ranges):
            imgp_rng = imgp[:,rng[0]:rng[1]]
            etatth_rngs.append( imgp_rng )
            omgeta_rngs.append( np.zeros([len(self.imgs), imgp_rng.shape[0]], dtype=np.float32) )
            omgtth_rngs.append( np.zeros([len(self.imgs), imgp_rng.shape[1]], dtype=np.float32) )
            omgeta_rngs[-1][0,:] = np.nansum(imgp_rng, 1)
            omgtth_rngs[-1][0,:] = np.nansum(imgp_rng, 0)        

        for ii in range(1,len(self.imgs)):
            imsum += 1*self.imgs[ii]
            imgp = warp_polar(1.*self.imgs[ii], center=q0_pos, radius=max_rad)
            etatth = etatth + imgp
            omgeta[ii,:] = np.nansum(imgp, 1)
            omgtth[ii,:] = np.nansum(imgp, 0)
            for ir, rng in enumerate(rad_ranges):
                imgp_rng = imgp[:,rng[0]:rng[1]]
                etatth_rngs[ir] = etatth_rngs[ir] + imgp_rng
                omgeta_rngs[ir][ii,:] = np.nansum(imgp_rng, 1)
                omgtth_rngs[ir][ii,:] = np.nansum(imgp_rng, 0)

        mask_rng = 0*imsum
        crds = np.mgrid[0:mask_rng.shape[1]:1, 0:mask_rng.shape[1]:1]
        r = np.sqrt( (crds[0,:,:]-q0_pos[0] )**2 + (crds[1,:,:]-q0_pos[1])**2 )
        for rng in rad_ranges:
            d = abs(r - (rng[0]+rng[1])/2)
            mask_rng[d<(rng[1]-rng[0])/2] = 1

        self.set_attr('projs', {'imsum' :  imsum,  'immax':  immax, 'q0_pos': q0_pos,
                                'etatth': etatth, 'omgtth': omgtth, 'omgeta': omgeta,
                                'ranges': rad_ranges,
                                'imsum_rngs' : imsum*mask_rng, 'immax_rngs': immax*mask_rng,
                                'etatth_rngs':etatth_rngs, 'omgtth_rngs':omgtth_rngs, 'omgeta_rngs':omgeta_rngs})

        del max_rad, imgp, etatth, omgeta, omgtth, imgp_rng, etatth_rngs, omgeta_rngs, omgtth_rngs
        return


    def plot(self):
        """Method to plot all the projections."""
        if type(self.projs) == dict:
            projs = self.projs
        else:
            projs = pickle.load( open(self.directory + self.name + "_projs.p", "rb"))
                          
        fig = plt.figure(figsize=(16, 10))

        sub = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
        im = sub.imshow(np.log10(projs['imsum']), interpolation='None')
        plt.title('Sum of images')
        divider = make_axes_locatable(sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        sub = fig.add_subplot(222)
        im = sub.imshow(np.log10(projs['etatth']), interpolation='None')
        plt.title('eta_tth projection')
        divider = make_axes_locatable(sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        sub = fig.add_subplot(223)
        im = sub.imshow(np.log10(projs['omgeta']), interpolation='None')
        plt.title('omg_eta projection')
        divider = make_axes_locatable(sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        sub = fig.add_subplot(224)
        im = sub.imshow(np.log10(projs['omgtth']), interpolation='None')
        plt.title('omg_tth projection')
        divider = make_axes_locatable(sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        plt.show()
        self.add_to_log('Writing file: '+self.directory+self.name+'_projs.png', False)
        fig.savefig(self.directory+self.name+'_proj.png')
        return

    
    
    def calculate_peaksearch_thrs(self):
        """Method to calculate thresholds from max projection image. Returns a list of thresholds."""
        from lmfit.models import LinearModel, GaussianModel, PseudoVoigtModel, LorentzianModel
        immax_inpolar, ptSettings = polarTransform.convertToPolarImage(
            self.projs['immax'],
            center=[self.geometry.y_center, self.geometry.z_center]) # max projection in polar coordinates

        immax_inpolar[immax_inpolar < 3] = np.nan
        tth_immax_profile = np.nanpercentile(immax_inpolar, 99.9, 0) # peaks for each tth
        tth_smoothed = scipy.ndimage.gaussian_filter1d(tth_immax_profile, 10)
        tth_pks_ind = scipy.signal.find_peaks(tth_smoothed, distance =10, height = 3)[0]
        tth_peaks_sorted = np.sort([tth_smoothed[ind] for ind in tth_pks_ind])

        x = np.linspace(0, len(tth_peaks_sorted), num = len(tth_peaks_sorted))
        y = np.asarray(tth_peaks_sorted)

        pseudo_voigt = PseudoVoigtModel(prefix = 'PseudoVoigtModel_')
        pars = pseudo_voigt.guess(y, x = x)
        lin_mod = LinearModel(prefix = 'Linear_')
        pars.update(lin_mod.make_params())
        mod = pseudo_voigt + lin_mod
        init = mod.eval(pars, x = x)
        out  = mod.fit(y, pars, x = x)

        diff = abs(scipy.ndimage.gaussian_filter1d(out.best_fit - y, 3))

        diff_pks = scipy.signal.find_peaks(diff, distance =3, height = 3)[0]
        if len(diff_pks) > 0:
            base_thr = 2*(3 + y[int(diff_pks[0]/2)])
        else:
            base_thr = 2*(3 + y[int(np.argmax(y)/2)])
        thrs = base_thr*np.asarray( [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] )
        thrs = np.asarray([t for t in thrs if t > 9 and t < np.max(self.projs['immax'])/2]) # reasonable range is [9 counts, max_intentsity/2] 
        self.set_attr('peaksearch_thrs', [int(np.round(t)) for t in thrs])
        return self.peaksearch_thrs
    
    
    def export_data(self, thr=None):
        """Export data (projections and compressed images) using a certain threshold."""
        self.add_to_log("Exporting data as: "+self.directory+self.name, False)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            self.add_to_log('Created directory: '+self.directory, False)
        
        if not thr or thr == 'auto': thr = min(self.thresholds)
    
        path = self.directory+self.name
        
        try:
            self.add_to_log('Writing file: '+path+f"_t{thr}.npz", False)
            imageseries.write(self.imgs,"dummy","frame-cache",cache_file=path+f"_t{thr}.npz", threshold=thr)
        except: self.add_to_log('Failed to write .npz file!', False)
        
        try:
            self.add_to_log('Writing file: '+path+"_bckg.tif", False)
            tifffile.imsave(path+"_bckg.tif", self.processing['bckg'])
        except: self.add_to_log('Failed to write *_bckg.tif file!', False)
        
        if os.path.exists(self.directory+'projs_ranges/'):
            del_old  = subprocess.call('rm '+self.directory+'projs_ranges/'+self.name+'*', shell=True)
            if del_old == 0: self.add_to_log(f"Deleted old files.", False)  
        else:
            os.makedirs(self.directory+'projs_ranges/')
            self.add_to_log('Created directory: '+self.directory+'projs_ranges/', False)
            
        path = self.directory+'projs_ranges/'+self.name
        try:
            self.add_to_log('Writing file: '+path+"_proj_immax.tif", False)
            tifffile.imsave(path+"_proj_immax.tif", self.projs['immax'])
        except: self.add_to_log('Failed to write *_proj_immax.tif file!', False)
        
        try:        
            self.add_to_log('Writing file: '+path+"_proj_imsum.tif", False)
            tifffile.imsave(path+"_proj_imsum.tif" , self.projs['imsum'])
        except: self.add_to_log('Failed to write *_proj_imsum.tif file!', False)

        try:            
            self.add_to_log('Writing file: '+path+"_proj_etatth.tif", False)
            tifffile.imsave(path+"_proj_etatth.tif", self.projs['etatth'])
        except: self.add_to_log('Failed to write *_proj_etatth.tif file!', False)

        try:            
            self.add_to_log('Writing file: '+path+"_proj_omgtth.tif", False)
            tifffile.imsave(path+"_proj_omgtth.tif", self.projs['omgtth'])
        except: self.add_to_log('Failed to write *_proj_omgtth.tif file!', False)

        try:            
            self.add_to_log('Writing file: '+path+"_proj_omgeta.tif", False)
            tifffile.imsave(path+"_proj_omgeta.tif", self.projs['omgeta'])
        except: self.add_to_log('Failed to write *_proj_omgeta.tif file!', False)
        
        
        try:
            tifffile.imsave(path+"_imsum_rngs.tif", self.projs['imsum_rngs'])
            tifffile.imsave(path+"_immax_rngs.tif", self.projs['immax_rngs'])
            for ir, rng in enumerate(self.projs['ranges']):
                tifffile.imsave(path+f"_etatth_rngs-{rng[0]}-{rng[1]}.tif", self.projs['etatth_rngs'][ir])
                tifffile.imsave(path+f"_omgtth_rngs-{rng[0]}-{rng[1]}.tif", self.projs['omgtth_rngs'][ir])
                tifffile.imsave(path+f"_omgeta_rngs-{rng[0]}-{rng[1]}.tif", self.projs['omgeta_rngs'][ir])
        except: self.add_to_log('Failed to write range projections *.tif files!', True)

        self.geometry.save_par( directory = self.directory,  par_file = self.name+'.par',  overwrite = True)
        self.geometry.save_yml( directory = self.directory,  yml_file = self.name+'.yml',  overwrite = True)
        self.geometry.save_poni(directory = self.directory, poni_file = self.name+'.poni', overwrite = True)
            
        return


    def file_series_object(self):
        """Method to create a file series object for FABLE in the rare case you need it."""
        import fabio
        order = list(range(len(self.imgs)))
        def frm(i):
            omega = 0.5*self.chunk['omegas'][0][0] + 0.5*self.chunk['omegas'][0][1]
            f = fabio.fabioimage.fabioimage( self.imgs[i], {'Omega': omega} )
            f.filename = self.chunk['filenames'][i]
            f.currentframe = i
            return f

        yield( frm(order[0]) ) # first
        for i in order: yield frm(i)
    
    
    def peaksearch(self, thresholds = [], spline_path = None):
        """Peaksearches the images in the class. Never used it."""
        if thresholds: self.set_attr('thresholds', thresholds)
        self.add_to_log(f'Running peaksearch on {len(self.imgs)} images using {len(self.thresholds)} thresholds...', False)
            
        if spline_path:
            os.system('cp '+spline_path+' '+self.directory+self.name+'.spline') # Copy spline file here
            self.geometry.set_attr('spline_file', self.name+'.spline')
        elif self.geometry.spline_file:
            spline_path = self.geometry.directory + self.geometry.spline_file
            os.system('cp '+spline_path+' '+self.directory+self.name+'.spline') # Copy spline file here
            self.geometry.set_attr('spline_file', self.name+'.spline')
        else:
            os.system('rm '+self.directory+self.name+'.spline') # Delete as not relevant
            self.geometry.set_attr('spline_file', None)        

        from ImageD11 import labelimage
        for i_thr, thr in enumerate(self.thresholds):
            peaksfile = open(self.directory+self.name+f'_peaks_t{thr}.flt', 'w')
            lio = labelimage.labelimage(self.imgs[0].shape, peaksfile)
            for i_img, img in enumerate(self.imgs):
                omg = np.mean(self.chunk['omegas'][i_img])
                lio.peaksearch(img, thr, omg)
                lio.mergelast()
            lio.finalise()
            peaksfile.close()
            self.add_to_log(self.directory+self.name+f'_peaks_t{thr}.flt', False)
        
        if self.geometry.spline_file:
            for thr in self.thresholds:
                flt_file = self.directory+self.name+f'_peaks_t{thr}.flt'
                apply_spline_to_fltfile(flt_file, flt_file, self.geometry.spline_file, self.projs['immax'].shape[0])
                self.add_to_log(f"For {flt_file} and threshold {thr} applied spline file:", self.geometry.spline_file)

        return


    def save_peaksearch_yaml(self, thresholds='auto', pix_tol=None, spline_path=None): # Configuration file for peaksearch
        """Creates a yaml file that is convenint to run peaksearcher."""
        if not thresholds:
            thresholds = self.peaksearch_thrs
        elif 'auto' in thresholds:
            thresholds = self.calculate_peaksearch_thrs()
        else:
            self.set_attr('peaksearch_thrs', thresholds)
        
        if not pix_tol: pix_tol = self.pix_tol
        
        if spline_path:
            os.system('cp '+spline_path+' '+self.directory+self.name+'.spline') # Copy spline file here
            self.geometry.set_attr('spline_file', self.name+'.spline')
        elif self.geometry.spline_file:
            if self.geometry.spline_file[0] == '/':
                os.system('cp '+self.geometry.spline_file+' '+self.directory+self.name+'.spline') # Probably it is some other directory so need to be copied here
            else:
                os.system('cp '+self.geometry.directory+self.geometry.spline_file+' '+self.directory+self.name+'.spline') # Probably it is some other directory so need to be copied here
            self.geometry.set_attr('spline_file', self.name+'.spline')
        elif os.path.exists(self.directory+self.name+'.spline'):
            os.system('rm '+self.directory+self.name+'.spline') # Delete as not relevant
        
        self.add_to_log('Writing file: '+self.directory+self.name+'_peaksearch.yaml', False)
        with open(self.directory+self.name+'_peaksearch.yaml', 'w') as f:
            f.write('# parameters for peaksearch')
            f.write('\nimage_dir: {}'.format(self.sweep['directory']))
            f.write('\nimage_ext: {}'.format(self.sweep['ext']))
            f.write('\nimage_stem: {} #Stem of images'.format(self.sweep['stem']))
            f.write('\nndigits: {}'.format(self.sweep['ndigits']))
            dig_part = self.chunk['filenames'][0].replace(self.sweep['stem'],'').replace(self.sweep['ext'],'')
            f.write('\nfirst_image: {}  # Index of first image'.format( int(dig_part) ))
            f.write('\nnbr_images: {}  # Number of images to peaksearch'.format(len(self.chunk['omegas'])))
            f.write('\nomegastep: {}'.format(self.sweep['omega_step']))
            f.write('\nstartomega: {}'.format(0.5*self.chunk['omegas'][0][0]+0.5*self.chunk['omegas'][0][1])) # Use middle omega.
            f.write('\ndark_image: {}'.format(self.directory+self.name+'_bckg.tif'))
            f.write('\nthresholds: {}'.format(thresholds))
            if spline_path:
                f.write('\nspline: {}'.format(self.directory+self.geometry.spline_file))
            f.write('\noutput_dir: {}'.format(self.directory))
            f.write('\nstem_out: {}'.format(self.name))
            f.write('\n#kwargs: \'--OmegaOverride\'#additional keyword aguments for peaksearch.py as one string')
            f.write('\n# parameters for merge_flt')
            if pix_tol:
                f.write('\npixel_tol: {} #minimum distance between peaks (pixels)'.format(pix_tol))
            f.write('\nmerged_name: \'{}\''.format(self.name+'.flt'))
        f.close()
        return self.directory+self.name+'_peaksearch.yaml'

    
    def run_peaksearcher(self, yaml_file='auto', use_imgs=False, use_temp_tifs=False, del_temp_tifs=True):
        """Wrapper for the ImageD11 peaksearch.py script."""
        if yaml_file == 'auto': yaml_file = self.save_peaksearch_yaml()
        with open(yaml_file) as f: pars = yaml.safe_load(f)
        if pars['stem_out'] == None: pars['stem_out'] = ''
        
        first_im = int(pars['first_image'])
        last_im  = int(pars['first_image']) + int(pars['nbr_images']) - 1
        path_out = os.path.join(pars['output_dir'], pars['stem_out']+'_peaks')
        if use_temp_tifs:
            path_inp = os.path.join(self.directory+self.name+"_temp/", pars['image_stem'])
            pars['image_ext'] = '.tif'
            pars['dark_image'] = None
            
            if os.path.exists(self.directory+self.name+"_temp/"):
                subprocess.call('rm '+self.directory+self.name+"_temp/*.tif", shell=True) # Delete old tif files
            else:
                os.makedirs(self.directory+self.name+"_temp/")
            
            self.add_to_log(f"Saving temporary images in "+self.directory+self.name+"_temp/", False)
            for ind, img in enumerate(self.imgs):
                dig_part = str(first_im+ind).zfill(pars['ndigits'])
                tifffile.imsave(path_inp+dig_part+pars['image_ext'], img)
        else:
            path_inp = os.path.join(pars['image_dir'], pars['image_stem'])

        # construct the command for peaksearch.py
        command = 'peaksearch.py -o {} -n {} '.format(path_out, path_inp)
        command+= '-F {} --ndigits {:d} '.format(pars['image_ext'], pars['ndigits'])
        command+= '-f {:d} -l {:d} '.format(first_im, last_im)
        command+= '-S {:.3f} -T {:.3f} -p Y'.format(pars['omegastep'], pars['startomega'])
        if pars['dark_image']: command+= ' -d {}'.format(pars['dark_image']) 
        for t in pars['thresholds']: command += ' -t {:d}'.format(t)
        if 'kwargs' in pars: command += ' {}'.format(pars['kwargs'])
        
        self.add_to_log('Running peaksearch in: '+self.directory+'\n'+command, False)
        print(command)
        if use_imgs:
            import time
            reallystart = time.time()
            try:
                from argparse import ArgumentParser
                parser = ArgumentParser()
                myparser = pyTSXRD,peaksearcher.get_options(parser)
                options , args = myparser.parse_known_args(command.split())
                options.file_series_object =  self.file_series_object()
                peaksearcher.peaksearch_driver(options, args)
            except:
                #if myparser is not None:
                    #myparser.print_help()
                #print("\n\n And here is the problem:\n")
                raise

            end = time.time()
            t = end-reallystart
            #print("Total time = %f /s" % ( t ))
        else:
            print(command.split())
            process = subprocess.run(command.split(), check=True,
                                     stdout=subprocess.PIPE, universal_newlines=True)
            self.add_to_log('Output:'+process.stdout, False)
            #print('Last line in the output:'+process.stdout.splitlines()[-1])
        self.set_attr('peaksearch_thrs', pars['thresholds'])
        
        if 'spline' in pars.keys():
            self.geometry.set_attr('spline_file', pars['spline'])
            for t in pars['thresholds']:
                flt_file = pars['output_dir']+pars['stem_out']+f'_peaks_t{t}.flt'
                os.system('cp '+flt_file+' '+flt_file.replace('.flt', '_nospline.flt')) #
                self.add_to_log(f'Saved '+flt_file.replace('.flt', '_nospline.flt')+' file (backup).')
                apply_spline_to_fltfile(flt_file, flt_file, pars['spline'], self.projs['immax'].shape[0])
                self.add_to_log(f"For {flt_file} and threshold {t} applied spline file:", pars['spline'])
                     
        if use_temp_tifs and del_temp_tifs:
            subprocess.call('rm -r '+self.directory+self.name+"_temp/", shell=True)
            self.add_to_log(f"Deleted temporary images.", False)
        return


    def run_peakmerger(self, yaml_file='auto', thresholds = 'auto'): 
        """Wrapper for the merge_flt.py script from ImageD11. Parameters are loaded from yaml file. Only specified thresholds are used."""
        try:
            self.geometry.save_par( directory = self.directory,  par_file = self.name+'.par',  overwrite = True)
        except:
            raise FileNotFoundError('Could not write *par file!')
                
        if yaml_file == 'auto': yaml_file = self.save_peaksearch_yaml()
        with open(yaml_file) as f: pars = yaml.safe_load(f)
        if pars['stem_out'] == None: pars['stem_out'] = ''

        if not thresholds:
            thresholds = self.peakmerge_thrs
        elif 'auto' in thresholds:
            thresholds = pars['thresholds']

        # Filtering peaks that have reasonable intensity and the number of pixels
        path_out = pars['output_dir'] + pars['stem_out']
        for thr in pars['thresholds']:
            I_min = 4*thr
            I_max = 8*thr if thr < max(pars['thresholds']) else 8000*thr
            PI = pyTSXRD.PeakIndexer(directory = pars['output_dir'])
            PI.load_flt(flt_file = pars['stem_out']+ f'_peaks_t{thr}.flt') # merged
            peaks_in_range = []
            for p in PI.peaks:
                if p['IMax_int'] > I_min:
                    if p['IMax_int'] < I_max:
                        if p['Number_of_pixels'] > 2:
                            peaks_in_range.append(p)
            PI.set_attr('peaks', peaks_in_range)
            PI.save_flt(flt_file = pars['stem_out']+ f'_peaks_cleaned_t{thr}.flt', overwrite = True)
            if len(peaks_in_range) < 1 and thr in thresholds:
                thresholds.remove(thr)       
        
        inp = os.path.join(pars['output_dir'], pars['stem_out']+'_peaks_cleaned')
        par_file  = self.geometry.directory+self.geometry.name+'.par'
        if 'merged_name' in pars:
            file_out = os.path.join(pars['output_dir'],pars['merged_name'])
        else:
            file_out = os.path.join(pars['output_dir'],pars['stem_out']+'.flt')
        # construct the command for merge_flt.py
        command = 'merge_flt.py {} {} {} {:d} '.format(par_file,inp,file_out,pars['pixel_tol'])
        command+= ('{:d} '*len(thresholds)).format(*thresholds)
        self.add_to_log('Merging flt files matching: '+inp+'\n'+command, False)
        process=subprocess.run(command.split(), check=True, stdout=subprocess.PIPE, universal_newlines=True)
        self.add_to_log('Output:'+process.stdout, False)
        #print('Last line in the output:'+process.stdout.splitlines()[-1])
        self.set_attr('peakmerge_thrs', thresholds)
        self.set_attr('pix_tol', pars['pixel_tol'])
        return


    def save_tifs(self): 
        """Method to save images as tif files."""
        path = self.directory+self.name+"_tifs/"
        self.add_to_log(f"Saving tifs to: "+path, False)
        if not os.path.exists(path):
            os.makedirs(path)
            self.add_to_log('Created directory: '+path, False)
        for i in range(0,len(self.imgs)):
            tifffile.imsave(path+"{:04d}.tif".format(i), np.float32(self.imgs[i]))
        self.add_to_log(f"Saved {len(self.imgs)} .tif files", False)        
        return
    
    
    def crop_imgs(self, roi):
        """Method to crop the images."""
        imgs = []
        for img in self.imgs:
            imgs.append(img[roi[0]:roi[1], roi[2]:roi[3]])
        self.set_attr('imgs', imgs)
        self.geometry.set_attr('y_center', self.geometry.y_center-roi[0])
        self.geometry.set_attr('z_center', self.geometry.z_center-roi[2])
        self.geometry.set_attr('dety_size', roi[1]-roi[0])
        self.geometry.set_attr('detz_size', roi[3]-roi[2])
        self.add_to_log(f"Cropped images according to roi: {roi}", False)        
        return
    
    
    def delete_tifs(self):
        """Deletes tiff files if they have been created by save_tifs method."""
        command = 'rm -r '+self.directory+self.name+"_tifs/"
        subprocess.call(command, shell=True)
        self.add_to_log(f"Deleted temporary .tif files", False)
        return
    
    
    def generate_PeakIndexer(self, directory = None, name = None):
        """Creates a PeakIndexer object needed for subsequent analysis."""
        if not directory: directory = self.directory
        if not name: name = self.name
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.add_to_log('Created directory: '+directory, True)
        PI = pyTSXRD.PeakIndexer(directory = directory)
        PI.load_flt(flt_file = name+'.flt') # merged
        PI.set_attr('sweepProcessor', self)
        PI.set_attr('geometry', copy.copy(self.geometry))
        PI.geometry.set_attr('omegasign', abs(self.geometry.omegasign))
#         PI.geometry.save_par(directory = directory, par_file = name+'.par', overwrite = True)
        self.add_to_log('Generated PeakIndexer in '+PI.directory+' named '+PI.name, False)
        return PI
    
    
    def dump_arrays(self):
        """Dumps the heavy arrays to files, so the object occupoies less memory."""
        np.save(self.directory+self.name+'_bckg.npy', self.processing['bckg'])
        self.add_to_log('Saved bckg to: ' + self.name+'_bckg.npy', False)
        self.processing['bckg'] = None
        
        pickle.dump( self.projs, open(self.directory + self.name + "_projs.p", "wb") )
        self.add_to_log('Saved projections to: ' + self.name+'_projs.p', False)
        self.projs = None
        return
        
    def load_arrays(self):
        """Loads the data back from the files that were created by dump_array methop."""
        self.processing['bckg'] = np.load(self.directory+self.name+'_bckg.npy')
        self.add_to_log('Loaded bckg from: ' + self.name+'_bckg.npy', False)
        
        self.projs = pickle.load( open(self.directory + self.name + "_projs.p", "rb") )
        self.add_to_log('Loaded projections to: ' + self.name+'_projs.p', False)
        return

def calculate_bckg(imgs, indices):
    """Calculates background using median image for a subset of images."""
    try: sub_set = np.asarray( [imgs[i] for i in indices] )
    except: raise ValueError(' - incorrect indices!')
    I = [img.mean() for img in imgs]
    norm  = np.max(I)/np.mean(I) # This corrects for non-uniformities in the sweep's total intensity profile
    return norm*np.median(sub_set,axis=0).astype(float32)


def apply_spline_to_fltfile(flt_file_in, flt_file_out, spline, sc_dim): 
    """MUST BE CHECKED BEFORE USING! Applies spline to the existing flt files and saves the updated peaks."""
    from ImageD11 import columnfile, blobcorrector
    if spline == 'perfect':
        cor = blobcorrector.perfect()
    else:
        cor = blobcorrector.correctorclass( spline )

    inc = columnfile.columnfile( flt_file_in )
    inc.s_raw = sc_dim - inc.s_raw
    for i in range( inc.nrows ):
        inc.sc[i], inc.fc[i] = cor.correct( inc.s_raw[i], inc.f_raw[i] )
    inc.sc = sc_dim - inc.sc
    inc.writefile( flt_file_out )