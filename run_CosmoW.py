#######################################################################
# This code was created by Alessandro Marins (University of Sao Paulo)
#
# Its public code
#
# Any question about your operation, please, 
# contact me in alessandro.marins@usp.br
#######################################################################


import os,sys
import json
import numpy as np
import argparse
import Cosmo_functions as cf
import Eisenstein_Hu_fits as EH
import creating_dir
from time import time,strftime, gmtime

###################################################################
# Check the python version and import configparser
###################################################################

if sys.version_info[0]==2:
	import ConfigParser
	config = ConfigParser.RawConfigParser()
elif sys.version_info[0]==3:
	import configparser
	config = configparser.ConfigParser()



###################################################################
# This part is for extracting information from parameters.ini file
###################################################################
timei       = time()
INI         = "parameters.ini"
name_params = os.path.join(os.getcwd(),INI)
config.read(name_params)
#Cosmology
Ob_h2          = config.getfloat("Cosmology","Obh2")
Oc_h2          = config.getfloat("Cosmology","Och2")
Ok_h2          = config.getfloat("Cosmology","Okh2")
h              = config.getfloat("Cosmology","h")
TCMB           = config.getfloat("Cosmology","TCMB")
Neff           = config.getfloat("Cosmology","Neff")
ns             = config.getfloat("Cosmology","ns")
As             = config.getfloat("Cosmology","As")
sigma8         = config.getfloat("Cosmology","sigma8")
sigma8_camb    = config.getfloat("Cosmology","sigma8_camb")
model_w        = config.get(     "Cosmology","model_w")
w0             = config.getfloat("Cosmology","w0")
wa             = config.getfloat("Cosmology","wa")
densities_unit = config.get(     "Cosmology","densities_unit")

#21cm
OmegaHI_model = config.get("21cm","OmegaHI_model")
biasHI_model  = config.get("21cm","biasHI_model")
unit_21cm     = config.get("21cm","unit_21cm")


#Power Spectrum
type_of_amplitude  = config.get(     "Power Spectrum","type_of_amplitude")
value_of_amplitude = config.get(     "Power Spectrum","value_of_amplitude")
kpivot             = config.getfloat("Power Spectrum","kpivot")
kmax               = config.getfloat("Power Spectrum","kmax")
kmin               = config.getfloat("Power Spectrum","kmin")
R_sigma            = config.getfloat("Power Spectrum","R_sigma")

#Angular Power Spectrum
lmin           = config.getint("Angular Power Spectrum","lmin")
lmax           = config.getint("Angular Power Spectrum","lmax")
window         = config.get(   "Angular Power Spectrum","window")

#IM telescope
n0              = config.getfloat(  "IM Telescope","n0")
Omega_Survey    = config.getfloat(  "IM Telescope","Omega_Survey")
V_Survey        = config.getfloat(  "IM Telescope","V_Survey")
Theta_FWHM      = config.getfloat  ("IM Telescope","Theta_FWHM")
V_pix           = config.getfloat(  "IM Telescope","V_pix")
T_sys           = config.getfloat(  "IM Telescope","T_sys")
n_f             = config.getint(    "IM Telescope","n_f")
t_obs           = config.getfloat(  "IM Telescope","t_obs")
use_z           = config.getboolean("IM Telescope","use_z")
z_min_Survey    = config.getfloat(  "IM Telescope","z_min_Survey")
z_max_Survey    = config.getfloat(  "IM Telescope","z_max_Survey")
freq_min_Survey = config.getfloat(  "IM Telescope","freq_min_Survey")
freq_max_Survey = config.getfloat(  "IM Telescope","freq_max_Survey")
n_bins          = config.getint(    "IM Telescope","n_bins")
D_antenna_diam  = config.getfloat(  "IM Telescope","D_antenna_diam")
fsky            = config.getfloat(  "IM Telescope","fsky")


#outputs
try:
	transfer_function  = json.loads(config.get( "outputs", "transfer_function"))
	for i in range(len(transfer_function)): 
		transfer_function[i]=str(transfer_function[i])
except:
	raise NameError

normalization_transfer_function = config.get(        "outputs","normalization_transfer_function")
transfer_function_txt            = config.getboolean("outputs","transfer_function_txt")
transfer_function_plot           = config.getboolean("outputs","transfer_function_plot")
path_output_transfer_function    = config.get(       "outputs","path_output_transfer_function")


power_spectrum             = config.getboolean("outputs","power_spectrum")
redshifts_plot             = config.get(       "outputs","redshifts_plot").split(",")
power_spectrum_txt         = config.getboolean("outputs","power_spectrum_txt")
power_spectrum_plot        = config.getboolean("outputs","power_spectrum_plot")
path_output_power_spectrum = config.get(       "outputs","path_output_power_spectrum")


angular_power_spectrum_21cm             = config.getboolean("outputs","angular_power_spectrum_21cm")
angular_power_spectrum_21cm_txt         = config.getboolean("outputs","angular_power_spectrum_21cm_txt")
angular_power_spectrum_21cm_plot        = config.getboolean("outputs","angular_power_spectrum_21cm_plot")
angular_power_spectrum_21cm_bins_plot   = config.get(       "outputs","angular_power_spectrum_21cm_bins_plot").split(",")
angular_power_spectrum_21cm_type_plot   = config.get(       "outputs","angular_power_spectrum_21cm_type_plot")
angular_power_spectrum_21cm_alm_map     = config.getboolean("outputs","angular_power_spectrum_21cm_alm_map")
path_output_angular_power_spectrum_21cm = config.get(       "outputs","path_output_angular_power_spectrum_21cm")


cosmic_variance      = config.getboolean("outputs","cosmic_variance")
cosmic_variance_plot = config.getboolean("outputs","cosmic_variance_plot")
shot_noise           = config.getboolean("outputs","shot_noise")
shot_noise_plot      = config.getboolean("outputs","shot_noise_plot")
foregrounds          = config.getboolean("outputs","foregrounds")
foregrounds_plot     = config.getboolean("outputs","foregrounds_plot")
poisson_noise        = config.getboolean("outputs","poisson_noise")
poisson_noise_plot   = config.getboolean("outputs","poisson_noise_plot")
thermal_noise        = config.getboolean("outputs","thermal_noise")
thermal_noise_plot   = config.getboolean("outputs","thermal_noise_plot")


redshifts_plot                        = creating_dir.remove_space_input(redshifts_plot)
angular_power_spectrum_21cm_bins_plot = creating_dir.remove_space_input(angular_power_spectrum_21cm_bins_plot)


###############################################################################
# You can modify any options in the parameters.ini file by the command terminal
###############################################################################

parser = argparse.ArgumentParser(description='Modify by the command terminal parameters in parameters.ini file')
#Cosmology
parser.add_argument('--Obh2'          , action = 'store', dest = 'Ob_h2'         , default = Ob_h2,          help = 'Baryon density parameter times h square')
parser.add_argument('--Och2'          , action = 'store', dest = 'Oc_h2'         , default = Oc_h2,          help = 'CDM density parameter times h square')
parser.add_argument('--Okh2'          , action = 'store', dest = 'Ok_h2'         , default = Ok_h2,          help = 'Curvature density parameter times h square')
parser.add_argument('--h'             , action = 'store', dest = 'h'             , default = h ,             help = 'h=H0/100. Where H0 is Hubble constant')
parser.add_argument('--TCMB'          , action = 'store', dest = 'TCMB'          , default = TCMB,           help = 'CMB temperature')
parser.add_argument('--Neff'          , action = 'store', dest = 'Neff'          , default = Neff,           help = 'Effective number of neutrino species')
parser.add_argument('--As'            , action = 'store', dest = 'As'            , default = As,             help = '')
parser.add_argument('--ns'            , action = 'store', dest = 'ns'            , default = ns,             help = '')
parser.add_argument('--sigma8'        , action = 'store', dest = 'sigma8'        , default = sigma8,         help = '')
parser.add_argument('--sigma8_camb'   , action = 'store', dest = 'sigma8_camb'   , default = sigma8_camb,    help = '')
parser.add_argument('--model_w'       , action = 'store', dest = 'model_w'       , default = model_w,        help = 'EoS model')
parser.add_argument('--w0'            , action = 'store', dest = 'w0'            , default = w0,             help = 'EoS')
parser.add_argument('--wa'            , action = 'store', dest = 'wa'            , default = wa,             help = 'EoS in CPL model')
parser.add_argument('--densities_unit', action = 'store', dest = 'densities_unit', default = densities_unit, help = 'Densities unit')
#21cm
parser.add_argument('--OmegaHI_model' , action = 'store', dest = 'OmegaHI_model' , default = OmegaHI_model,  help = '')
parser.add_argument('--biasHI_model'  , action = 'store', dest = 'biasHI_model'  , default = biasHI_model,   help = '')
parser.add_argument('--unit_21cm'     , action = 'store', dest = 'unit_21cm'     , default = unit_21cm,      help = '')
#PowerSpectrum
parser.add_argument('--type_of_amplitude' , action = 'store', dest = 'type_of_amplitude' , default = type_of_amplitude,  help = '')
parser.add_argument('--value_of_amplitude', action = 'store', dest = 'value_of_amplitude', default = value_of_amplitude, help = '')
parser.add_argument('--kpivot'            , action = 'store', dest = 'kpivot'            , default = kpivot,             help = '')
parser.add_argument('--kmax'              , action = 'store', dest = 'kmax'              , default = kmax,               help = 'Maximum k value')
parser.add_argument('--kmin'              , action = 'store', dest = 'kmin'              , default = kmin,               help = 'Minimum k value')
parser.add_argument('--R_sigma'           , action = 'store', dest = 'R_sigma'           , default = R_sigma,            help = '')
#AngularPowerSpectrum
parser.add_argument('--lmin'      , action = 'store', dest = 'lmin'        , default = lmin,        help = 'Minimum multipolo')
parser.add_argument('--lmax'      , action = 'store', dest = 'lmax'        , default = lmax,        help = 'Maximum multipolo')
parser.add_argument('--window'    , action = 'store', dest = 'window'      , default = window,      help = 'type of window function: battye, tophat, gaussian, zero or one.')
#IM Telescope
parser.add_argument('--n0'             , action = 'store', dest = 'n0'             , default = n0,             help = ' sources comovel numerical density')
parser.add_argument('--Omega_Survey'   , action = 'store', dest = 'Omega_Survey'   , default = Omega_Survey,   help = '')
parser.add_argument('--V_Survey'       , action = 'store', dest = 'V_Survey'       , default = V_Survey,       help = '')
parser.add_argument('--Theta_FWHM'     , action = 'store', dest = 'Theta_FWHM'     , default = Theta_FWHM,     help = '')
parser.add_argument('--V_pix'          , action = 'store', dest = 'V_pix'          , default = V_pix,          help = '')
parser.add_argument('--T_sys'          , action = 'store', dest = 'T_sys'          , default = T_sys,          help = '')
parser.add_argument('--n_f'            , action = 'store', dest = 'n_f'            , default = n_f,            help = '')
parser.add_argument('--t_obs'          , action = 'store', dest = 't_obs'          , default = t_obs,          help = '')
parser.add_argument('--use_z'          , action = 'store', dest = 'use_z'          , default = use_z,          help = '')
parser.add_argument('--z_min_Survey'   , action = 'store', dest = 'z_min_Survey'   , default = z_min_Survey,   help = '')
parser.add_argument('--z_max_Survey'   , action = 'store', dest = 'z_max_Survey'   , default = z_max_Survey,   help = '')
parser.add_argument('--freq_min_Survey', action = 'store', dest = 'freq_min_Survey', default = freq_min_Survey,help = '')
parser.add_argument('--freq_max_Survey', action = 'store', dest = 'freq_max_Survey', default = freq_max_Survey,help = '')
parser.add_argument('--n_bins'         , action = 'store', dest = 'n_bins'         , default = n_bins,         help = 'number of bins')
parser.add_argument('--D_antenna_diam' , action = 'store', dest = 'D_antenna_diam' , default = D_antenna_diam, help = '')
parser.add_argument('--fsky'           , action = 'store', dest = 'fsky'           , default = fsky          , help = '')
#Outputs
parser.add_argument('--transfer_function'                , action = 'store', dest = 'transfer_function'               , default = transfer_function,                help = 'models of transfer functions')
parser.add_argument('--normalization_transfer_function'  , action = 'store', dest = 'normalization_transfer_function' , default = normalization_transfer_function, help = '')
parser.add_argument('--transfer_function_txt'            , action = 'store', dest = 'transfer_function_txt'           , default = transfer_function_txt,            help = '')
parser.add_argument('--transfer_function_plot'           , action = 'store', dest = 'transfer_function_plot'          , default = transfer_function_plot,           help = '')
parser.add_argument('--path_output_transfer_function'    , action = 'store', dest = 'path_output_transfer_function'   , default = path_output_transfer_function,    help = '')

parser.add_argument('--power_spectrum'                , action = 'store', dest = 'power_spectrum'               , default = power_spectrum,                help = 'models of power spectrum')
parser.add_argument('--redshifts_plot'                , action = 'store', dest = 'redshifts_plot'               , default = redshifts_plot,                help = 'models of power spectrum')
parser.add_argument('--power_spectrum_txt'            , action = 'store', dest = 'power_spectrum_txt'           , default = power_spectrum_txt,            help = '')
parser.add_argument('--power_spectrum_plot'           , action = 'store', dest = 'power_spectrum_plot'          , default = power_spectrum_plot,           help = '')
parser.add_argument('--path_output_power_spectrum'    , action = 'store', dest = 'path_output_power_spectrum'   , default = path_output_power_spectrum,    help = '')

parser.add_argument('--angular_power_spectrum_21cm'                , action = 'store', dest = 'angular_power_spectrum_21cm'               , default = angular_power_spectrum_21cm,                help = 'models of 21cm angular power spectrum')
parser.add_argument('--angular_power_spectrum_21cm_txt'            , action = 'store', dest = 'angular_power_spectrum_21cm_txt'           , default = angular_power_spectrum_21cm_txt,            help = '')
parser.add_argument('--angular_power_spectrum_21cm_plot'           , action = 'store', dest = 'angular_power_spectrum_21cm_plot'          , default = angular_power_spectrum_21cm_plot,           help = '')
parser.add_argument('--angular_power_spectrum_21cm_bins_plot'      , action = 'store', dest = 'angular_power_spectrum_21cm_bins_plot'     , default = angular_power_spectrum_21cm_bins_plot,      help = '')
parser.add_argument('--angular_power_spectrum_21cm_type_plot'      , action = 'store', dest = 'angular_power_spectrum_21cm_type_plot'     , default = angular_power_spectrum_21cm_type_plot,      help = '')
parser.add_argument('--angular_power_spectrum_21cm_alm_map'        , action = 'store', dest = 'angular_power_spectrum_21cm_alm_map'       , default = angular_power_spectrum_21cm_alm_map,        help = '')
parser.add_argument('--path_output_angular_power_spectrum_21cm'    , action = 'store', dest = 'path_output_angular_power_spectrum_21cm'   , default = path_output_angular_power_spectrum_21cm,    help = '')

parser.add_argument('--cosmic_variance'      , action = 'store', dest = 'cosmic_variance'      , default = cosmic_variance,      help = '')
parser.add_argument('--cosmic_variance_plot' , action = 'store', dest = 'cosmic_variance_plot' , default = cosmic_variance_plot, help = '')
parser.add_argument('--shot_noise'           , action = 'store', dest = 'shot_noise'           , default = shot_noise,           help = '')
parser.add_argument('--shot_noise_plot'      , action = 'store', dest = 'shot_noise_plot'      , default = shot_noise_plot,      help = '')
parser.add_argument('--foregrounds'          , action = 'store', dest = 'foregrounds'          , default = foregrounds,          help = '')
parser.add_argument('--foregrounds_plot'     , action = 'store', dest = 'foregrounds_plot'     , default = foregrounds_plot,     help = '')
parser.add_argument('--poisson_noise'        , action = 'store', dest = 'poisson_noise'        , default = poisson_noise,        help = '')
parser.add_argument('--poisson_noise_plot'   , action = 'store', dest = 'poisson_noise_plot'   , default = poisson_noise_plot,   help = '')
parser.add_argument('--thermal_noise'        , action = 'store', dest = 'thermal_noise'        , default = thermal_noise,        help = '')
parser.add_argument('--thermal_noise_plot'   , action = 'store', dest = 'thermal_noise_plot'   , default = thermal_noise_plot,   help = '')


arguments = parser.parse_args()
###############################################################################
# Variables
###############################################################################
#Cosmology
Ob_h2          = float(arguments.Ob_h2)
Oc_h2          = float(arguments.Oc_h2)
Ok_h2          = float(arguments.Ok_h2)
h              = float(arguments.h)
TCMB           = float(arguments.TCMB)
Neff           = float(arguments.Neff)
As             = float(arguments.As)
ns             = float(arguments.ns)
sigma8         = float(arguments.sigma8)
sigma8_camb    = float(arguments.sigma8_camb)
model_w        = str(arguments.model_w)
w0             = float(arguments.w0)
wa             = float(arguments.wa)
densities_unit = str(arguments.densities_unit)

#21cm
OmegaHI_model = str(arguments.OmegaHI_model)
biasHI_model  = str(arguments.biasHI_model)
unit_21cm     = str(arguments.unit_21cm)

#PowerSpectrum
type_of_amplitude   = str(arguments.type_of_amplitude)
values_of_amplitude = str(arguments.value_of_amplitude)
kpivot              = float(arguments.kpivot)
kmax                = float(arguments.kmax)
kmin                = float(arguments.kmin)
R_sigma             = float(arguments.R_sigma)

#AngularPowerSpectrum
lmin           = int(arguments.lmin)
lmax           = int(arguments.lmax)
window         = str(arguments.window)

#IM Telescope
n0              = float(arguments.n0)
Omega_Survey    = float(arguments.Omega_Survey)
V_Survey        = float(arguments.V_Survey)
Theta_FWHM      = float(arguments.Theta_FWHM)
V_pix           = float(arguments.V_pix)
T_sys           = float(arguments.T_sys)
n_f             = int(arguments.n_f)
t_obs           = float(arguments.t_obs)
use_z           = str(arguments.use_z)
z_min_Survey    = float(arguments.z_min_Survey)
z_max_Survey    = float(arguments.z_max_Survey)
freq_min_Survey = float(arguments.freq_min_Survey)
freq_max_Survey = float(arguments.freq_max_Survey)
n_bins          = int(arguments.n_bins)
D_antenna_diam  = float(arguments.D_antenna_diam)
fsky            = float(arguments.fsky)

#Outputs
transfer_function                = list(arguments.transfer_function)
normalization_transfer_function  = str(arguments.normalization_transfer_function)
transfer_function_txt            = bool(arguments.transfer_function_txt)
transfer_function_plot           = bool(arguments.transfer_function_plot)
path_output_transfer_function    = str(arguments.path_output_transfer_function)

power_spectrum             = bool(arguments.power_spectrum)
redshifts_plot             = list(arguments.redshifts_plot)
power_spectrum_txt         = bool(arguments.power_spectrum_txt)
power_spectrum_plot        = bool(arguments.power_spectrum_plot)
path_output_power_spectrum = str(arguments.path_output_power_spectrum)

angular_power_spectrum_21cm             = bool(arguments.angular_power_spectrum_21cm)
angular_power_spectrum_21cm_txt         = bool(arguments.angular_power_spectrum_21cm_txt)
angular_power_spectrum_21cm_plot        = bool(arguments.angular_power_spectrum_21cm_plot)
angular_power_spectrum_21cm_bins_plot   = list(arguments.angular_power_spectrum_21cm_bins_plot)
angular_power_spectrum_21cm_type_plot   = str(arguments.angular_power_spectrum_21cm_type_plot)
angular_power_spectrum_21cm_alm_map     = bool(arguments.angular_power_spectrum_21cm_alm_map)
path_output_angular_power_spectrum_21cm = str(arguments.path_output_angular_power_spectrum_21cm)


cosmic_variance      = bool(arguments.cosmic_variance)
cosmic_variance_plot = bool(arguments.cosmic_variance_plot)
shot_noise           = bool(arguments.shot_noise)
shot_noise_plot      = bool(arguments.shot_noise_plot)
foregrounds          = bool(arguments.foregrounds)
foregrounds_plot     = bool(arguments.foregrounds_plot)
poisson_noise        = bool(arguments.poisson_noise)
poisson_noise_plot   = bool(arguments.poisson_noise_plot)
thermal_noise        = bool(arguments.thermal_noise)
thermal_noise_plot   = bool(arguments.thermal_noise_plot)

redshifts_plot                          = np.asarray(redshifts_plot)
angular_power_spectrum_21cm_bins_plot   = np.asarray(angular_power_spectrum_21cm_bins_plot)
transfer_function                       = np.asarray(transfer_function)



###############################################################################
# Building "params" 
###############################################################################

params    = {"Ob_h2":Ob_h2,"Oc_h2":Oc_h2,"Ok_h2":Ok_h2,"h":h,
             "TCMB":TCMB,"Neff":Neff,"As":As,"ns":ns, 
             "sigma8":sigma8,"sigma8_camb":sigma8_camb,
             "w0":w0,"wa":wa,"model_w":model_w,
             "densities_unit":densities_unit,
             "type_of_amplitude":type_of_amplitude,"value_of_amplitude":value_of_amplitude,
             "kpivot":kpivot,"kmax":kmax,"kmin":kmin,"lmin":lmin,"lmax":lmax,
             "R_sigma":R_sigma,
             "transfer_function":transfer_function, "normalization_transfer_function":normalization_transfer_function}

paramSurvey = {"OmegaHI_model":OmegaHI_model,"biasHI_model":biasHI_model,"unit_21cm":unit_21cm,
               "n0":n0,"Omega_Survey":Omega_Survey,"V_Survey":V_Survey,"Theta_FWHM":Theta_FWHM,
               "V_pix":V_pix, "T_sys":T_sys,"n_f":n_f,"t_obs":t_obs,
               "lmin":lmin,"lmax":lmax,"kmax":kmax,"kmin":kmin,
               "window":window,
               "use_z":use_z,"z_min_Survey":z_min_Survey,"z_max_Survey":z_min_Survey,
               "freq_min_Survey":freq_min_Survey,"freq_max_Survey":freq_max_Survey,
               "n_bins":n_bins,"D_antenna_diam":D_antenna_diam,
                "transfer_function":transfer_function}           

'''
paramsOutput = {"transfer_function":transfer_function, "normalization_transfer_function":normalization_transfer_function, "transfer_function_txt":transfer_function_txt, "transfer_function_plot":transfer_function_plot,
               "path_output_transfer_function":path_output_transfer_function,
               "power_spectrum":power_spectrum,"power_spectrum_txt":power_spectrum_txt,"power_spectrum_plot":power_spectrum_plot,
               "path_output_power_spectrum":path_output_power_spectrum,
               "angular_power_spectrum_21cm":angular_power_spectrum_21cm,"angular_power_spectrum_21cm_txt":angular_power_spectrum_21cm_txt, "angular_power_spectrum_21cm_plot":angular_power_spectrum_21cm_plot,
               "angular_power_spectrum_21cm_alm_map":angular_power_spectrum_21cm_alm_map, "path_output_angular_power_spectrum_21cm":path_output_angular_power_spectrum_21cm
               }
'''               
      
###############################################################################
# Other Variables
###############################################################################
As    = As*1e-9
h2    = h**2
Om_h2 = Ob_h2+Oc_h2
Or_h2 = cf.Orh2(params)
Od_h2 = h2 - (Om_h2 + Ok_h2 + Or_h2)

n0   *= h**3
del_freq         = (freq_max_Survey-freq_min_Survey)/n_bins 
gamma_max_Survey = 2*15*np.pi/360
bin_l            = np.pi/gamma_max_Survey

params['As']      = As
params['h2']      = h2
params['Om_h2']   = Om_h2
params['Or_h2']   = Or_h2
params["Og_h2"]   = cf.Ogammah2(params)
params['Od_h2']   = Od_h2
params['bin_l']   = bin_l
paramSurvey['n0'] = n0
paramSurvey['gamma_max_Survey'] = gamma_max_Survey



###############################################################################
###############################################################################
###############################################################################
# Beginning the program
###############################################################################
###############################################################################
###############################################################################



print("###############################################################################")
print("Beginning the program")
print("###############################################################################")
print("\n\n")


###############################################################################
# Plots configuration
###############################################################################

if transfer_function_plot or power_spectrum_plot or angular_power_spectrum_21cm_plot:
	import matplotlib.pyplot as plt
	import matplotlib
	from matplotlib import rc
	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	rc('text', usetex=True)
	font = {'family': 'normal','weight': 'bold','size': 22}
	matplotlib.rc('font', **font)


###############################################################################
# Outputs directory
###############################################################################

if transfer_function_plot or transfer_function_txt or (power_spectrum_plot or power_spectrum_txt)*power_spectrum or (angular_power_spectrum_21cm_plot or angular_power_spectrum_21cm_txt)*angular_power_spectrum_21cm:
	print("Checking if the outputs path exists...")
	path_outputs = creating_dir.verification_dir("outputs",os.getcwd())
	print(" ".join(("Outputs path is:",path_outputs)))
	print("\n")

for i, name in enumerate(transfer_function):
	if not i:
		names = "".join(("#",name))
	else:
		names = np.hstack((names,"".join(("#",name))))
		
###############################################################################
# Building the Transfer Function
###############################################################################
print(" ".join(("Transfer Functions:",str(transfer_function))))
print("Building the Transfer Functions...")

k                 = np.logspace(np.log10(kmin),np.log10(kmax),num=1e4)
transfer_function = np.asarray(transfer_function)
TF                = EH.TF_vector(k,params)


if normalization_transfer_function == "1":
	print("Transfer Functions normalization: 1")
	if len(np.where(transfer_function=="camb")[0])>0:
		if len(np.where(transfer_function=="EH")[0])>0:
			alpha = TF[np.where(transfer_function=="EH")[0][0],0]/TF[np.where(transfer_function=="camb")[0][0],0]
			TF[np.where(transfer_function=="camb")[0][0],:]*=alpha
		else:
			par   = params
			par["transfer_functions"] = "EH"
			TFEH  = EH.TF_vector(k,par)
			alpha = TFEH[0]/TF[np.where(transfer_function=="camb")[0][0],0]
			TF[np.where(transfer_function=="camb")[0][0],:]*=alpha
			del alpha,par,TFEH
			
	elif len(np.where(transfer_function=="camb")[0])==0 and type_of_amplitude=="camb":
		par        = params
		par['transfer_functions']="camb"
		TFcamb     = EH.TF_vector(k,par)
		alpha_camb = TF[0,0]/TFcamb[0]
		params["alpha_camb"] = alpha_camb
		del TFcamb,par

elif normalization_transfer_function=="camb":
	print("Transfer Functions normalization: camb")
	if (len(np.where(transfer_function=="EH")[0])+len(np.where(transfer_function=="noBAO")[0])+len(np.where(transfer_function=="nobaryons")[0])+len(np.where(transfer_function=="BBKS")[0]))>0:
		loc_camb = np.where(transfer_function=="camb")[0][0]
		len_TF   = np.shape(transfer_function)[0]
		for  i in range(len_TF):
			if i==loc_camb:
				pass
			else:
				TF[i,:] *= (TF[loc_camb,0]/TF[i,0])
else:
	pass

if transfer_function_txt or transfer_function_plot:
	
	if transfer_function_txt:
		print("Saving the transfer functions data...")
		path = creating_dir.verification_dir("datas",path_outputs)
		
		if path_output_transfer_function=="standard":
			np.savetxt(os.path.join(path, "transfer_function.txt"), np.vstack((names,TF.T)) , delimiter = "   ", fmt="%s")
		else:
			try:
				np.savetxt(path_output_transfer_function, np.vstack((names,TF.T)) , delimiter = "   ", fmt="%s")
			except:
				raise NameError(" ".join(("There is no such path:",path_output_transfer_function)))
	
	if transfer_function_plot:
		print("Saving the transfer functions plot...")
		for i,model in enumerate(transfer_function):
			plt.loglog(k,TF[i,:],label=model)
		plt.xlabel(r"$k\left[\textrm{Mpc}^{-1}\right]$")
		plt.ylabel(r'Transfer Function')
		plt.legend(loc="best")
		if path_output_transfer_function=="standard":
			path = creating_dir.verification_dir("plots",path_outputs)
			plt.savefig(os.path.join(path,"transfer_function.png"),bbox_inches = "tight")
		else:
			plt.savefig(os.path.join(os.getcwd(),path_output_power_spectrum,".png"),bbox_inches = "tight")
		plt.tight_layout()
		plt.show()

###############################################################################
# Building the Power Spectrum
###############################################################################

if power_spectrum:
	print("\n\nPower Spectrum: " + str(transfer_function))
	print("Redshift(s): " + str(redshifts_plot))
	print("Building the 3D matter Power Spectrum...")
	

	len_z = len(redshifts_plot)
	if len_z==1:
		
		PK = EH.Pk(k,params,TF)
		del TF
		if power_spectrum_plot:
			print("Saving the power spectrum plot...")
			path = creating_dir.verification_dir("plots",path_outputs)
			for i,model in enumerate(transfer_function):
				plt.loglog(k,PK[i,:],label=model)
			plt.xlabel(r"$k\left[\textrm{Mpc}^{-1}\right]$")
			plt.ylabel(r'$P(k)\ \left[\textrm{Mpc}^3\right]$ ')
			plt.legend(loc="best")
			if path_output_transfer_function=="standard":
				plt.savefig(os.path.join(path,"power_spectrum.png"),bbox_inches = "tight")
			else:
				plt.savefig(os.path.join(path_output_power_spectrum,"power_spectrum.png"),bbox_inches = "tight")
			plt.tight_layout()
			plt.show()
			
			del names

		if power_spectrum_txt:
			print("Saving the power spectrum data...")
			path = creating_dir.verification_dir("datas",path_outputs)
			
			if path_output_power_spectrum=="standard":
				np.savetxt(os.path.join(path, "power_spectrum.txt"), np.vstack((names,PK.T)) , delimiter = "   ", fmt="%s")
			else:
				try:
					np.savetxt(path_output_power_spectrum, np.vstack((names,PK.T)) , delimiter = "   ", fmt="%s")
				except:
					raise NameError("There is no such path: " + path_output_power_spectrum)

			
	elif len_z>1:
		PKz = EH.Pkz(redshifts_plot,params,TF)
		del TF,names
		len_PK = len(transfer_function)
		
		for i in range(len_z):
			for j in range(len_PK):
				if not i + j:
					names = ' '.join(("#",transfer_function[j],"in","z =",str(redshifts_plot[i])))
				else:
					names = np.hstack((names,' '.join(("#",transfer_function[j],"in","z =",str(redshifts_plot[i])))))
		print(len_PK)

		
		if power_spectrum_plot:
			ki=0
			print("Saving the power spectrum plot...")
			path = creating_dir.verification_dir("plots",path_outputs)
			for i in range(len_z):
				for j in range(len_PK):
					label_PKz = ' '.join((transfer_function[j],"in","z =",str(redshifts_plot[i])))
					plt.loglog(k,PKz[ki,:],label=label_PKz)
					ki+=1
			
			plt.xlabel(r"$k\left[\textrm{Mpc}^{-1}\right]$")
			plt.ylabel(r'$P(k)\ \left[\textrm{Mpc}^3\right]$ ')
			if (len_z+len_PK)>6:
				plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize = 'x-small')
			else:
				plt.legend(loc="best", fontsize = 'x-small')
			if path_output_transfer_function=="standard":
				plt.savefig(os.path.join(path,"power_spectrum.png"),bbox_inches = "tight")
			else:
				plt.savefig(os.path.join(path_output_power_spectrum,"power_spectrum.png"),bbox_inches = "tight")
			plt.tight_layout()
			plt.show()				
			del ki,names,len_z,len_PK

		
		if power_spectrum_txt:
			print("Saving the power spectrum data...")
			path = creating_dir.verification_dir("datas",path_outputs)
			
			if path_output_power_spectrum=="standard":
				np.savetxt(os.path.join(path, "power_spectrum.txt"), np.vstack((names,PKz.T)) , delimiter = "   ", fmt="%s")
			else:
				try:
					np.savetxt(path_output_power_spectrum, np.vstack((names,PKz.T)) , delimiter = "   ", fmt="%s")
				except:
					raise NameError("There is no such path: " + path_output_power_spectrum)


###############################################################################
# Building the 21cm Angular Power Spectrum
###############################################################################

if angular_power_spectrum_21cm:
	import z_vector as zv
	import HI_functions as hi
	from scipy.interpolate import interp1d

	del_l     = 1
	zvector   = zv.vec_z(paramSurvey) 
	zz        = np.linspace(0,z_max_Survey+1,endpoint=True,num=1e3)
	lvector   = np.arange(lmin,lmax+1,del_l)
  
  
	print(" ".join(("\n\n21cm Angular Power Spectrum:",str(transfer_function))))
	print(" ".join(("Redshift range:"," - ".join((str(z_min_Survey),str(z_max_Survey))))))
	print("Saving redshifts data...")
	print("Saved.")
	path = creating_dir.verification_dir("datas",path_outputs)
	np.savetxt(os.path.join(path,"redshifts.txt"),zvector)
	print("Building the 21cm Angular Power Spectrum...")

	if not power_spectrum:
		PK = EH.Pk(k,params,TF)

	
	Dz        = cf.D_z(zz,params,True)
	Dz        = interp1d(zz,Dz)
	
	for i in range(len(transfer_function)):
		for j in range(len(zvector)-1):
			z_bin     = np.array([zvector[j],zvector[j+1]])
			cl_i_delz = EH.CL21_Limber(lvector,z_bin, k, PK[i], Dz, params, paramSurvey)
			print("Calculated (model,bin): ({},{})".format(transfer_function[i],j+1))
			print("(z_{},z_{}): {}".format(j,j+1,z_bin))
			if not i+j:
				CL = cl_i_delz
			else:
				CL = np.vstack((CL,cl_i_delz))
	del cl_i_delz,z_bin,zz,Dz,del_l
	
	
	if angular_power_spectrum_21cm_txt:
		print("Saving the 21cm angular power spectrum data...")
		path = creating_dir.verification_dir("datas",path_outputs)
		CLL = np.vstack((lvector,CL))

		for i,tf in enumerate(transfer_function):
			for j in range(n_bins):
				if not i+j:
					names = "#l"
					names = np.hstack((names,"".join(("#",tf," in bin ",str(j+1)))))
				else:
					names = np.hstack((names,"".join(("#",tf," in bin ",str(j+1)))))
		
		if path_output_power_spectrum=="standard":			
			np.savetxt(os.path.join(path, "21cm_angular_power_spectrum.txt"), np.vstack((names,CLL.T)) , delimiter = "   ", fmt="%s")
		
		else:
			try:
				np.savetxt(path_output_power_spectrum, np.vstack((names,CLL.T)) , delimiter = "   ", fmt="%s")
			
			except:
				raise NameError("There is no such path: " + path_output_power_spectrum)
		del CLL

	
				
	if angular_power_spectrum_21cm_plot:
		
		for i in range(len(transfer_function)):
			if not i:
				index = angular_power_spectrum_21cm_bins_plot-1	
			else:
				index = np.hstack((index,i*n_bins+(angular_power_spectrum_21cm_bins_plot-1)))
		index = index.astype(int)
		
		if n_bins*len(angular_power_spectrum_21cm_bins_plot)>1:
			CL = CL[index,:]
		
			print("Saving the power spectrum plot...")
			path = creating_dir.verification_dir("plots",path_outputs)
			
			ki=0
			if angular_power_spectrum_21cm_type_plot=="dl":
				for i,model in enumerate(transfer_function):
					for j, bin_z in enumerate(angular_power_spectrum_21cm_bins_plot):
						label_CL = ' '.join((model,"in","bin =",str(int(bin_z))))
						plt.loglog(lvector,lvector*(lvector+1)*CL[ki,:]/2./np.pi,label=label_CL)
						ki+=1

			elif angular_power_spectrum_21cm_type_plot=="cl":
				for i,model in enumerate(transfer_function):
					for j, bin_z in enumerate(angular_power_spectrum_21cm_bins_plot):
						label_CL = ' '.join((model,"in","bin =",str(int(bin_z))))
						plt.loglog(lvector,CL[ki,:],label=label_CL)
						ki+=1
			
			else:
				raise NameError("There is no such " + str(angular_power_spectrum_21cm_type_plot))
				
		else:
			CL = CL[:]
		
			print("Saving the power spectrum plot...")
			path = creating_dir.verification_dir("plots",path_outputs)
			
			ki=0
			if angular_power_spectrum_21cm_type_plot=="dl":
				for i,model in enumerate(transfer_function):
					for j, bin_z in enumerate(angular_power_spectrum_21cm_bins_plot):
						label_CL = ' '.join((model,"in","bin =",str(int(bin_z))))
						plt.loglog(lvector,lvector*(lvector+1)*CL/2./np.pi,label=label_CL)
						ki+=1

			elif angular_power_spectrum_21cm_type_plot=="cl":
				for i,model in enumerate(transfer_function):
					for j, bin_z in enumerate(angular_power_spectrum_21cm_bins_plot):
						label_CL = ' '.join((model,"in","bin =",int(str(bin_z))))
						plt.loglog(lvector,CL,label=label_CL)
						ki+=1
			
			else:
				raise NameError("There is no such " + str(angular_power_spectrum_21cm_type_plot))			
			
			
			
		plt.xlabel(r"$\ell$")
		plt.ylabel(r'$C_{\ell}[\mu\textrm{K}^2]$ ')
		if (len(angular_power_spectrum_21cm_bins_plot)+len(transfer_function))>6:
			plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize = 'x-small')
		else:
			plt.legend(loc="best", fontsize = 'x-small')
		if path_output_angular_power_spectrum_21cm=="standard":
			plt.savefig(os.path.join(path,"21cm_angular_power_spectrum.png"),bbox_inches = "tight")
		else:
			try:    plt.savefig(os.path.join(path_output_angular_power_spectrum_21cm,"21cm_angular_power_spectrum.png"),bbox_inches = "tight")
			except: raise NameError
		plt.tight_layout()
		plt.show()				
	

###############################################################################
# Noises
###############################################################################
if (cosmic_variance or shot_noise or foregrounds or poisson_noise or thermal_noise)*angular_power_spectrum_21cm:
	print("\nBuilding noise signals for angular power spectrum...")
	import noises as no
	
	path = creating_dir.verification_dir("datas",path_outputs)
	
	#cosmic variance
	if cosmic_variance:
		print("cosmic variance")
		matrixCL = n_bins*len(angular_power_spectrum_21cm_bins_plot)>1
		CV = no.cosmic_variance(lvector,CL,fsky,matrixCL)
		CV = np.vstack((lvector,CV))
		no.write_noise("cosmic_variance",path,CV,paramSurvey)
		

	#shot noise
	if shot_noise:
		SN = no.shot_noise()
		try:
			noise = np.vstack((noise,SN))
		except:
			noise = SN
		
	#foregrounds		
	if foregrounds:
		FG = no.foregrounds()
		try:
			noise = np.vstack((noise,FG))
		except:
			noise = FG

	#poisson
	if poisson_noise:
		PN = no.poisson_noise()
		try:
			noise = np.vstack((noise,PN))
		except:
			noise = PN

	#thermal noise
	if thermal_noise:
		TN = np.thermal_noise()
		try:
			noise = np.vstack((noise,TN))
		except:
			noise = TN

	
	
	
	
###############################################################################
# Resume file: Cosmological parameters, program time, module time,...
###############################################################################

###############################################################################
# Finish
###############################################################################
print("\nEnd program.")
