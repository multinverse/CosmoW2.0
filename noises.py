import numpy as np
import os

def cosmic_variance(l,cl,fsky,matrixCL):
	
	if matrixCL:
		CV = np.empty((np.shape(cl)[0],np.shape(cl)[1]))
		for i in range(np.shape(cl)[0]):
			CV[i,:] = np.sqrt(2/(2*l + 1)/fsky)*cl[i,:]
		#CV = np.vstack(())
	else:
		CV = np.sqrt(2/(2*l + 1)/fsky)*cl
	
	return CV
		
def shot_noise():
	return None
	
def foregrounds():
	return None

def poisson_noise():
	return None

def thermal_noise():
	return None


def write_noise(name,path,data,paramSurvey):
	pathout = os.path.join(path,'.'.join((name,"txt")))
	for i,tf in enumerate(paramSurvey['transfer_function']):
		for j in range(paramSurvey['n_bins']):
			if not i+j:
				names = "#l"
				names = np.hstack((names,"".join(("#",tf,"_",name," noise in bin ",str(j+1)))))
			else:
				names = np.hstack((names,"".join(("#",tf,"_",name," noise in bin ",str(j+1)))))
	np.savetxt(pathout, np.vstack((names,data.T)) , delimiter = "   ", fmt="%s")
