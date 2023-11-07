import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
import vector
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict
import pickle
from distributed.diagnostics.plugin import UploadDirectory
import os
from plot_utils import adjust_plot
import matplotlib.pyplot as plt
from utils import Histfit
import correctionlib

from distributed.diagnostics.plugin import UploadDirectory
from qcd_processor_lib import QCDProcessor
from processor_utils import *

def runner(testing = True, year = 2017, era = "2017", prependstr = 'root://xcache/', nworkers = 2  ):
    if era == "2016":
        jerfile = "Summer20UL16_JRV3_MC_PtResolution_AK4PFchs.txt"
        filename = 'samples/flatPU_JMENano_2016.txt'
    if era == '2016APV':
        jerfile = "Summer20UL16APV_JRV3_MC_PtResolution_AK4PFchs.txt"    
        filename = 'samples/flatPU_JMENano_2016APV.txt'
    if era == '2017':
        jerfile = "Summer20UL17_JRV1_MC_PtResolution_AK4PFchs.txt"
        filename = 'samples/flatPU_JMENano_2017.txt'
    if era == '2018':
        jerfile = "Summer20UL18_JRV1_MC_PtResolution_AK4PFchs.txt"
        filename = 'samples/flatPU_JMENano_2018.txt'

    
    fileset = {}
    
    eras = [era]
    for era in eras:
        with open(filename) as f:
            files = [prependstr + i.rstrip() for i in f.readlines() if i[0] != '#']
            fileset[era] =  files
            
            
    if testing == True:
        fileset[era] = fileset[era][:1]
        exe_args = {
        "skipbadfiles": True,
        "schema": NanoAODSchema,
        "workers":1}
        
        hists = processor.run_uproot_job(
                        fileset,
                        treename="Events",
                        processor_instance=QCDProcessor(),
                        executor=processor.iterative_executor, #.futures_executor,
                        executor_args=exe_args,chunksize=10000,
                        maxchunks=1
                    )
    else:
        exe_args = {
        "skipbadfiles": True,
        "schema": NanoAODSchema,
        "workers":nworkers}

        hists = processor.run_uproot_job(
                        fileset,
                        treename="Events",
                        processor_instance=QCDProcessor(),
                        executor=processor.iterative_executor, #.futures_executor,
                        executor_args=exe_args,
                    )
    return hists