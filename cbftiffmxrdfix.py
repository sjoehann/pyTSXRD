#!/usr/bin/python3
"""
module/script to fix compatibility issues for CBF files created by ngMultiXRD4343 (prior to version 0.8.4) 
or for TIFF files using CFA photometric interpretation (as created by ngMultiXRD4343 via ASAP::O).


If your software uses the 'fabio' module to read CBF/TIFF files then simply import this module first (before importing 'fabio') 
to work around the issues regarding the CBF/TIFF format, e.g.:
    
    ...
    ...
    import cbfmxrdfix
    import fabio
    ...
    ...
    ...    
    # let's read a CBF now
    img = fabio.open('sweep_022272.cbf')
    

To actually convert the CBF/TIFF files created by ngMultiXRD4343 (prior to version 0.8.4) into proper ones, you can run the script like so:
    
    python3  cbftiffmxrdfix.py  -o destination_directory   sweep_058948.cbf   sweep_022272   run1234/sweep_*.cbf

where 'destination_directory' must not exist, it will contain the converted CBF files.
    
    
"""


import os, sys, tempfile
import re
import numpy as NP
assert NP.little_endian, "sorry...this only works on little-endian architectures"
import hashlib, base64
import io

import fabio
_fabioopen=fabio.open
def _fixedopen(filename, frame=None):
    """ 
    Open an image; if the image is a CBF file created by ngMultiXRD4343 it will be corrected
    for fabio to process it.

    It returns a FabioImage-class instance which can be used as a context
    manager to close the file at the termination.
    """
    if filename.lower().endswith(('.tif','.tiff')):
        data=tiffmxrdfix(filename).rewrite()
    else:   
        data=cbfmxrdfix(filename).rewrite()
    if fabio.version_info.major > 0 or fabio.version_info.minor >= 12:
        return _fabioopen(io.BytesIO(data), frame)
    else:
        fd,fname = tempfile.mkstemp(suffix=b'.cbf')       
        os.write(fd, data)
        os.close(fd)
        res=_fabioopen(fname.decode(), frame)
        os.unlink(fname)
        return res
fabio.open = _fixedopen


_header_end_mark = b'\x0C\x1A\x04\xD5'
_CBFVERSION = b'VERSION 1.5, created by ngMultiXRD4343 <V0.8.4'


class cbfmxrdfix(object):
    def __init__(self, fname):
        self.content = open(fname,"rb").read()
        self.ismxrd=(not self.content.startswith(b"###CBF") 
                     and b"X-Binary-Size-Padding: 0\r\r\n" in self.content 
                     and b"Content-Type: application / octet - stream; " in self.content)
        self.binsize=int(re.search(rb'X-Binary-Size:\s*"?(\d+)"?', self.content).groups()[0])
        binstart=self.content.find(_header_end_mark)+len(_header_end_mark)
        self.header=self.content[:binstart-len(_header_end_mark)].splitlines()
        self.binportion = bytearray(self.content[binstart:binstart+self.binsize])
      
    def rewrite(self, outfname=None):
        if not self.ismxrd:
            out = self.content
        else:                
            out=bytearray()
            binportion=self._recompfast()
            md5_hash = base64.b64encode(hashlib.md5(binportion).digest())
            header = self.header
            for i,h in enumerate(header):
                header[i] = header[i].rstrip()
                if h.startswith(b"Content-Type:"):
                    header[i] = b"Content-Type: application/octet-stream;"
                elif h.startswith(b"X-Binary-Size-Padding: 0"):            
                    header[i] = b"X-Binary-Size-Padding: 1\r\n"
                elif h.startswith(b"Content-Transfer - Encoding"):
                    header[i] = b'Content-Transfer-Encoding: BINARY'
                elif h.startswith(b"Content-MD5:") and md5_hash != None:
                    header[i] = b'Content-MD5: ' + md5_hash
                elif h.startswith(b'X-Binary-Size:') and md5_hash != None:
                    header[i] = b'X-Binary-Size: %u' % len(binportion)
            header.insert(0, b'###CBF: ' + _CBFVERSION)        
            out += b"\r\n".join(header) + _header_end_mark
            out += binportion + b'\0' + b'\r\n--CIF-BINARY-FORMAT-SECTION----\r\n;\r\n'
        if outfname is None:
            return out
        else:
            open(outfname, "wb").write(out)   

    def _recompfast(self):
        data=bytearray(self.binportion)
        out=bytearray()
        dlen=len(data)
        idx=0
        tok16,tok32=0x80,0x8000
        while idx < dlen:
            nidx=data.find(tok16, idx)
            if nidx < 0:
                out += data[idx:]
                break
            else:
                out += data[idx:nidx+1]
                val = NP.frombuffer(data[nidx+1:nidx+3], dtype=NP.int16).astype(NP.int32)[0]
                if abs(val) < tok32:
                    out += data[nidx+1:nidx+3]
                else:
                    out += NP.uint16(tok32).tobytes()
                    out += NP.int32(val).tobytes()
                idx = nidx+3
        return out


_TIFF_MAGIC = 0x2a4949
_TIFF_CFA_PHOTOMETRIC_TAG = b"\6\1"+b"\3\0"+b"\1\0\0\0"+b"\x23\x80"
_TIFF_OLDSTYLE_PHOTOMETRIC_TAG = b"\6\1"+b"\3\0"+b"\1\0\0\0"+b"\x01\x00"

class tiffmxrdfix(object):
    def __init__(self, fname):
        self.content = open(fname,"rb").read()
        id,ptr=NP.frombuffer(self.content[:8], dtype=NP.uint32)
        self.ismxrd = False
        istiff = (id == _TIFF_MAGIC)  # magic for TIFF
        if istiff: 
            self.tagptr = self.content.find(b"\6\1"+b"\3\0"+b"\1\0\0\0"+b"\x23\x80")
            self.ismxrd = self.tagptr > -1
      
    def rewrite(self, outfname=None):
        if not self.ismxrd:
            out = self.content
        else:                
            out=bytearray()
            out+=self.content[:self.tagptr]
            out+=_TIFF_OLDSTYLE_PHOTOMETRIC_TAG
            out+=self.content[self.tagptr+len(_TIFF_OLDSTYLE_PHOTOMETRIC_TAG):]
        if outfname is None:
            return out
        else:
            open(outfname, "wb").write(out)   





if __name__ == "__main__":
    usage = 'usage:\n\t{prog} -o <destination_dir> <cbf-file> [another cbf-file]...'
    assert len(sys.argv) >= 4, usage
    assert sys.argv[1] == '-o', usage
    destdir = sys.argv[2]
    try:
        os.makedirs(destdir)
    except:
        raise IOError("unable to create destination directory")
    for f in sys.argv[3:]:
        fabio.open(f).write(os.path.join(destdir, os.path.basename(f)))
    
    

    
    





