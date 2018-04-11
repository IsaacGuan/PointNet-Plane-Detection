import h5py
import numpy as np
from plyfile import PlyData, PlyElement

filenames = [line.rstrip() for line in open("filelist", 'r')]
labels = [line.rstrip() for line in open("labels", 'r')]

f = h5py.File("./hdf5_data/data.h5", 'w')

a_label = np.zeros((len(filenames), 1), dtype = np.uint8)
a_data = np.zeros((len(filenames), 2048, 3))
a_pid = np.zeros((len(filenames), 2048), dtype = np.uint8)	

for i in range(0, len(filenames)):
	a_label[i] = labels[i]
	plydata = PlyData.read("./ply/" + filenames[i] + ".ply")
	piddata = [line.rstrip() for line in open("./points_label/" + filenames[i] + ".seg", 'r')]
	for j in range(0, 2048):
		a_data[i, j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]
		a_pid[i,j] = piddata[j]

label = f.create_dataset("label", data = a_label)
data = f.create_dataset("data", data = a_data)
pid = f.create_dataset("pid", data = a_pid)
