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
import re

from distributed.diagnostics.plugin import UploadDirectory
from qcd_processor_lib import QCDProcessor
from processor_utils import *

def runner(testing = True, eras = ["2017"], prependstr = 'root://xcache/', nworkers = 2 , client = None ):

    
    fileset = {}
    
    #datasets = ["/RunIISummer20UL17NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/","/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/"]
    for era in eras:
        if era == '2017':
            filename = 'samples/flatPU_JMENano_2017.txt'
            metadata = '/RunIISummer20UL17NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/'
        elif era == '2018':
            filename = 'samples/flatPU_JMENano_2018.txt'
            metadata = '/RunIISummer20UL18NanoAODv9/QCD_Pt-15to7000_TuneCP5_Flat2018_13TeV_pythia8/NANOAODSIM/'
        else:
            print("era is:" + era)
            print("This is Unknown era")
        with open(filename) as f:
            files = [prependstr + i.rstrip() for i in f.readlines() if i[0] != '#']
            fileset[metadata] =  files
            
            
    if testing == True:
        fileset[list(fileset.keys())[0]] = fileset[list(fileset.keys())[0]][:1]
        
        if client == None:
            exe_args = {
            "skipbadfiles": True,
            "schema": NanoAODSchema,
            "workers":1}

            hists = processor.run_uproot_job(
                            fileset,
                            treename="Events",
                            processor_instance=QCDProcessor(),
                            executor=processor.futures_executor, #.futures_executor,
                            executor_args=exe_args,chunksize=1000000,
                            maxchunks=5
                        )
        else:
            exe_args = {
                "client": client,
                "skipbadfiles": True,
                "schema": NanoAODSchema,
                "align_clusters": True
            }
            hists = processor.run_uproot_job(
                fileset,
                treename="Events",
                processor_instance=QCDProcessor(),
                executor=processor.dask_executor,
                executor_args=exe_args,

                maxchunks=10, chunksize = 100000
                  )
    else:
        if client == None:
            exe_args = {
            "skipbadfiles": True,
            "schema": NanoAODSchema,
            "workers":nworkers}

            hists = processor.run_uproot_job(
                            fileset,
                            treename="Events",
                            processor_instance=QCDProcessor(),
                            executor=processor.iterative_executor, #.iterative_executor,#.futures_executor,
                            executor_args=exe_args,
                        )
            
            
        else:
            # exe_args = {
            #     "client": client,
            #     "skipbadfiles": True,
            #     "schema": NanoAODSchema,
            #     "align_clusters": True
            # }
            # hists = processor.run_uproot_job(
            #     fileset,
            #     treename="Events",
            #     processor_instance=QCDProcessor(),
            #     executor=processor.dask_executor,
            #     executor_args=exe_args
            #       )
            executor = processor.DaskExecutor(client=client)
            run = processor.Runner(executor=executor,
                        schema= NanoAODSchema,
                        savemetrics=False
                      )

            hists = run(fileset, "Events", processor_instance=QCDProcessor())
            
    
    return hists