from PhaseRetrieveTools import saveVISP3D
import numpy as np
import glob,sys

path = sys.argv[1]

print 'reading {}'.format(path)
found_nm = np.loadtxt(path,delimiter=',',skiprows=1)

newpath = path.replace('csv','3d')
print 'saving {}'.format(newpath)
saveVISP3D(found_nm,newpath,convert_um2nm=False)
