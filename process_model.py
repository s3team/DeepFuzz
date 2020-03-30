import h5py
import numpy as numpy

filename = "s2s.h5"
f = h5py.File(filename, 'r')

group = f["model_weights"]
for key in group.keys():
	print(key)