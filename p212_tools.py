single_separator = "--------------------------------------------------------------"
double_separator = "=============================================================="

### FUNCTION TO LOAD META DATA FROM P21.2 .LOG FILE
def load_p212_log(path):
    #print(single_separator+'\nReading p212 log file:', path)
    p212_log = {'file': path.split('/')[-1]}
    p212_log['directory'] = path.replace(p212_log['file'], '').replace('../', '')
    
    with open(path, "r") as f:
        
        p212_log['entries'] = []
        p212_log['data_and_time'] = f.readline()[1:-1]
        
        for line in f:
            
            if line[0] == '#':
                if 'crosshead' in line and 'position' in line:
                    p212_log['crosshead_position'] = float(line.split()[2])
                elif 'sweep' in line.split()[0]:
                    p212_log['command'] = line[1:-1]
                elif 'Detector' in line:
                    p212_log['det_num'] = (line.split()[1])
                elif 'timestamp' in line:
                    titles = line[1:].split()
            else:
                words = line.split()
                if len(words) == len(titles):
                    v = [float(v) for v in words]
                    p212_log['entries'].append( dict(zip(titles,v)) )
    f.close()
    
    loads = [ent['load'] for ent in p212_log['entries']]
    p212_log['pressure'] = sum(loads) / len(loads)
    #for k in p212_log.keys():
        #if k != 'entries': print(k, '=', p212_log[k])
    #print('In total', len(p212_log['entries']), 'log entries with keys:')
    #if p212_log['entries']: print([k for k in p212_log['entries'][0].keys()] )    
        
    return p212_log


### FUNCTION TO LOAD META DATA FROM P21.2 .FIO FILE
def load_p212_fio(path):
    #print(single_separator+'\nReading p212 fio file:', path)
    p212_fio = {'file': path.split('/')[-1]}
    p212_fio['directory'] = path.replace(p212_fio['file'], '').replace('../', '')
    
    with open(path, "r") as f:        
        p212_fio['entries'  ] = []
        p212_fio['detectors'] = []
        
        for line in f:
            if line[:-1] == '%c': 
                p212_fio['command'] = f.readline()[:-1]
            if 'acquisition started' in line:
                p212_fio['data_and_time'] = line[:-1].split('acquisition started at')[-1]
            if 'channel ' in line and ': Perk' in line:
                p212_fio['detectors'].append( int(line.split()[1][:-1]) )
            if 'channel1_exposure' == line.split()[0]:
                p212_fio['exposure'] = float(line[:-1].split()[-1])
            if 'energy' == line.split()[0]:
                p212_fio['energy']= float(line[:-1].split()[-1])
                
            if 'idrx1' == line.split()[0]:
                p212_fio['idrx1'] = float(line[:-1].split()[-1])            
            if 'idry1' == line.split()[0]:
                p212_fio['idry1'] = float(line[:-1].split()[-1])
            if 'idrz1' == line.split()[0]:
                p212_fio['idrz1'] = float(line[:-1].split()[-1])
                
            if 'idrx2' == line.split()[0]:
                p212_fio['idrx2'] = float(line[:-1].split()[-1])            
            if 'idry2' == line.split()[0]:
                p212_fio['idry2'] = float(line[:-1].split()[-1])
            if 'idrz2' == line.split()[0]:
                p212_fio['idrz2'] = float(line[:-1].split()[-1])
                
            if 'idtx1' == line.split()[0]:
                p212_fio['idtx1'] = float(line[:-1].split()[-1])
            if 'idty1' == line.split()[0]:
                p212_fio['idty1'] = float(line[:-1].split()[-1])
            if 'idtz1' == line.split()[0]:
                p212_fio['idtz1'] = float(line[:-1].split()[-1])
            if 'idtx2' == line.split()[0]:
                p212_fio['idtx2'] = float(line[:-1].split()[-1])
            if 'idty2' == line.split()[0]:
                p212_fio['idty2'] = float(line[:-1].split()[-1])
            if 'idtz2' == line.split()[0]:
                p212_fio['idtz2'] = float(line[:-1].split()[-1])
                
            if line[:-1] == '%d':
                titles = []
                types  = []
                
                while True:
                    words = f.readline().split()
                    if words[0] == 'Col':
                        titles.append( ' '.join(words[2:-1]) )
                        if  words[-1] == 'INTEGER':
                            types.append(1)
                        elif words[-1] == 'DOUBLE':
                            types.append(0.1)
                        else:
                            types.append('s')
                    else:
                        break
                        
                while True:
                    if len(words) == len(titles):
                        words = ['nan' if w in ['<no-data>', 'None'] else w for w in words]
                        v = [ type(types[i])(words[i]) for i in range(len(titles)) ]
                        p212_fio['entries'].append( dict(zip(titles,v)) )
                        words = f.readline().split()
                    else:
                        break
                        
    f.close()
    #for k in p212_fio.keys():
        #if k != 'entries': print(k, '=', p212_fio[k])
    #print('In total', len(p212_fio['entries']), 'log entries with keys:')
    #if p212_fio['entries']: print([k for k in p212_fio['entries'][0].keys()] )
    
    return p212_fio

def parse_p212_command(command):
    command = command.replace("='", "=")
    res = {'slow':{}, 'fast':{}, 'sweep':{}, 'directory':None, 'file_stem':None, 'file_ext':None, 'exposure':None}
    to_drop = ['[', ']', ',', '\'', '--']
    for c in to_drop: command  = command.replace(c,' ')
    words = command.split()

    if words[0] == 'sweepwrap':
        # Examples:
        # sweepwrap 75.50 235.50 640 0.10 newMgAl103    superz    5.98    6.73    6
        # sweepwrap 90.00 270.00 720 0.10 test_MgCaZn4    superz    6.25    6.55    3     supery  -0.8  0.8  9   dark  line
        sweep_command    = ['idrz1'] + words[1:5]
        res['directory'] = "/gpfs/current/raw/" + words[5]
        if len(words) < 11:
            fast_command = words[6:10]
        else:
            slow_command = words[6:10]
            fast_command = words[10:14]

    elif words[0] == 'supersweep':
        # Examples:
        # supersweep idtz2 10.375 10.865 50 idrz1 0.0 360.0 1:720/0.10 fdir="/gpfs/current/raw/copper1/part_2/before_000_%D" fname="before_.cbf" slaves=234 asapo=1 4
        fast_command      = words[1:5]
        sweep_command     = words[5:8] + words[8].split('/')
        fdir  = [w for w in words if 'fdir'  in w][0].split('=')[1].replace('\"', '')
        fname = [w for w in words if 'fname' in w][0].split('=')[1].replace('\"', '')
        res['directory'] = fdir
        res['file_ext' ] = '.' + fname.split('.')[-1]
        res['file_stem'] = fname.replace(res['file_ext'], '')
        
    elif words[0] == 'fastsweep':
        # Examples:
        # fastsweep idrz1 90.0 270.0 1:720/0.100,fdir="/gpfs/current/raw/test_MgCaZn2/48/y_0.000_z_6.250/%D",fname="sweep_.cbf",asapo=1 4
        # fastsweep idrz1 359.0 1.0 1:716/0.10,fdir="/gpfs/current/raw/Nb_sweepfull_5/Nb_sweepfull/0169_1",fname="sweep0169_.cbf",asapo=1 2:716/0.10,fdir="/gpfs/current/raw/Nb_sweepfull_5/Nb_sweepfull/0169_2",fname="sweep0169_.cbf",asapo=1 4
        sweep_command     = words[1:4] + words[4].split('/')
        fdir  = [w for w in words if 'fdir'  in w][0].split('=')[1].replace('\"', '')
        fname = [w for w in words if 'fname' in w][0].split('=')[1].replace('\"', '')
        res['directory'] = fdir
        res['file_ext' ] = '.' + fname.split('.')[-1]
        res['file_stem'] = fname.replace(res['file_ext'], '')     
        if len(words) > 10: res['directory'] = res['directory'][:-1]+'%D'
        
    elif words[0] == 'fastsweep2':
        # Examples:
        # fastsweep idrz1 90.0 270.0 1:720/0.100,fdir="/gpfs/current/raw/test_MgCaZn2/48/y_0.000_z_6.250/%D",fname="sweep_.cbf",asapo=1 4
        # fastsweep idrz1 359.0 1.0 1:716/0.10,fdir="/gpfs/current/raw/Nb_sweepfull_5/Nb_sweepfull/0169_1",fname="sweep0169_.cbf",asapo=1 2:716/0.10,fdir="/gpfs/current/raw/Nb_sweepfull_5/Nb_sweepfull/0169_2",fname="sweep0169_.cbf",asapo=1 4
        sweep_command     = words[1:4] + [w for w in words if ':' in w and '/' in w][0].split('/')
#         print('SWEEP_COMMAND:', sweep_command)
#         print('WORDS:', words)
        fdir = words[words.index('datadir')+1].replace('\"', '')
        
        try:
            fname = [w for w in words if 'fname' in w][0].split('=')[1].replace('\"', '')
        except:
            fname = "frame_.cbf"
        res['directory'] = fdir
        res['file_ext' ] = '.' + fname.split('.')[-1]
        res['file_stem'] = fname.replace(res['file_ext'], '')     
        if len(words) > 10: res['directory'] = res['directory']+'/Varex_%D'
    
    try:
        res['slow']['motor' ]  =   str(slow_command[0])
        res['slow']['start' ]  = float(slow_command[1])
        res['slow']['end'   ]  = float(slow_command[2])
        res['slow']['points']  =   int(slow_command[3])
        res['slow']['step'  ]  = ( res['slow']['end'] - res['slow']['start'] ) / ( res['slow']['points'] - 1 )
    except: pass
    
    try:
        res['fast']['motor' ]  =   str(fast_command[0])
        res['fast']['start' ]  = float(fast_command[1])
        res['fast']['end'   ]  = float(fast_command[2])
        res['fast']['points']  =   int(fast_command[3])
        res['fast']['step'  ]  = ( res['fast']['end'] - res['fast']['start'] ) / ( res['fast']['points'] - 1 )
    except: pass    
    
    try:
        res['sweep']['motor' ] =   str(sweep_command[0])
        res['sweep']['start' ] = float(sweep_command[1])
        res['sweep']['end'   ] = float(sweep_command[2])
        res['sweep']['points'] =   int(sweep_command[3].split(':')[-1]) + 1 # 1:720
        res['sweep']['step'  ] = ( res['sweep']['end'] - res['sweep']['start'] ) / ( res['sweep']['points'] - 1 )   
        res['exposure']        = float(sweep_command[4])
    except: pass
    
    return res