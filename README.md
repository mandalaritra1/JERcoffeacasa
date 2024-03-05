# JER From QCD datasets

Coffea implementation of finding JER using QCD datasets. 


## Instructions

This is designed to run in coffea-casa or LPC. Follow the instruction from [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue?tab=readme-ov-file ) to setup the environment. 
The coffea processor can be run from `qcd-processor-v4.ipynb`. It defaults to the LPC setup. To run in coffea-casa locally, change the `prependstr = 'root://xcache/'` and set `client = None`. 

The JER plots can be produced in `plotting.ipynb`. The range of interest in rho (or eta) can be changed by modifying `n_rho_min` and `n_rho_max`.

