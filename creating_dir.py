import numpy as np
import os,sys

def verification_dir(name,path):
	if not os.path.isdir(path):
		os.mkdir(path)
	else:
		pass
	if not os.path.isdir(os.path.join(path,name)):
		print("Directory "+ name + " does not exist.")
		os.mkdir(os.path.join(path,name))
		print("Directory "+ name + " created.")
	else:
		print("Directory "+ name + " exists.")
		pass
		
	return os.path.join(path,name)

def remove_space_input(var):
	var = np.char.strip(var)
	return np.asarray(var.astype(np.float))
	
