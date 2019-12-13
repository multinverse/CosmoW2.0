import numpy as np

def vec_z(params):
	import units_conversions as un
	nbins       = params['n_bins']
	z_vector    = np.zeros(int(nbins+1))
	freq        = np.zeros(int(nbins+1))
	len_z       = len(z_vector)
	l_vector    = np.arange(params["lmin"],params["lmax"],1)
	len_l       = len(l_vector)
	del_freq    = (params["freq_max_Survey"] - params["freq_min_Survey"])/nbins
	z_vector[0] = params["z_min_Survey"]
	for n in range(len_z): 
		z_vector[n] = (un.unit("freq_21cm_MHz")/(params["freq_min_Survey"] + n*del_freq))-1.
	z_vector = np.flip(z_vector,0)
	return z_vector
	
def vec_names(params):
	nbins = params['n_bins']
	for i in range(nbins):
		if    i==0:names = ['bin1.dat']
		else: names.append('bin'+str(i+1)+'.dat')
	return names



