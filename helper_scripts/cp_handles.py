import os
from shutil import copyfile
fs = os.listdir('handles/')
fs.sort()
for f in fs:
	ind = os.path.basename(os.path.normpath(f))
	print(ind)
	assert ind[-4:]=='.npy', ind[-4:]
	ind  = ind[0:8]
	assert ind.isdigit()
	copyfile(f'handles/{f}', f'/mnt/session_space/star_32/star_clean_processed/{ind}/handles.npy')	
