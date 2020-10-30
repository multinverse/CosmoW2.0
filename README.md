# CosmoW2.0

% This code was created by Alessandro Marins (University of Sao Paulo)
% Any question about your operation, please, 
% contact me in alessandro.marins@usp.br


This code is a public code and is a continuation of CosmoW, code that generates TF (transfer function), PK (power spectrum) and CL(angular power spectrum) of different types.

You can use it to generate different transfer function and spectra: BBKS, Eisenstein & Hu, No BAO, No Baryons and CAMB. It can also generate with these spectra 21cm angular spectrum.

All parameters are in the file parameters.ini. To execute just use:

> python3 run_CosmoW.py

All parameters can be changed at the terminal using "--", the parameter name and the value. For example, to change the value of baryons for 0.01:

> python3 run_CosmoW.py --obh2 0.01
