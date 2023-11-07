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
from processor_utils import *

class QCDProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        frac_axis = hist.axis.Regular(300, 0, 2.0, name="frac", label=r"Fraction")
        ptgen_axis = hist.axis.Variable([200,260,350,460,550,650,760,13000], name="ptgen", label=r"p_{T,RECO} (GeV)")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        pt_axis = hist.axis.Variable([10 ,  11  , 12 ,  13  , 14  , 15 ,  17,
       20  , 23  , 27   ,30  , 35   ,40  , 45 ,  57 ,  72  , 90  , 120 ,  150,
       200  , 300 ,  400   ,550 ,  750 ,  1000 ,  1500  , 2000 ,  2500  , 3000,
       3500 ,  4000  ,  5000   ,10000], name="pt", label=r"$p_{T}$ [GeV]") #erased 4000 and 5000

        
        
        
        
        pileup_axis = hist.axis.Variable([0, 10, 20, 30, 40, 50, 60, 70, 80],name = "pileup", label = r"$\mu$" )     
        pileup_fine_axis = hist.axis.Regular(30, 0, 40, name = 'pileup_fine', label = r"$\mu$")
        
        rho_axis = hist.axis.Variable( [0, 7.47, 13.49, 19.52, 25.54, 31.57, 37.59, 90], 
                                      name = 'rho', label = r'$\rho$')
        rho_fine_axis = hist.axis.Regular(30, 0, 30, name = 'rho_fine', label = r"$\rho$")
        
        
        #eta_axis = hist.axis.Regular(15, -4,4, name = "eta", label = r"$eta$")
        # eta_axis = hist.axis.Variable([0, 0.261, 0.522, 0.783,  1.044, 1.305, 1.566, 1.74, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853,
        #                               2.964, 3.139, 5],name = "eta", label = r"$\eta$")
        #eta_axis = hist.axis.Variable([-5.191, -3.839, -3.489, -3.139, -2.964, -2.853, -2.65, -2.5, -2.322,-2.172,-2.043, -1.93, -1.74, -1.566,-1.305,-1.044 ,-0.783 ,-0.522, -0.261, 0, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.74, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.853,], name = "eta", label = r"$\eta$")
        
        #eta_axis = hist.axis.Variable([0, 1.305,  2.5, 2.65, 2.853,
                                        #5.191],name = "eta", label = r"$\eta$")
        
        #eta_axis = hist.axis.Variable([ 0, 0.5, 0.8, 1.1, 1.3, 1.7, 1.9, 2.1, 2.3, 2.5, 2.8, 3, 3.2, 4.7],name = "eta", label = r"$\eta$")
        
        eta_axis = hist.axis.Variable([ 0, 0.261, 0.522, 0.783, 1.044, 1.305, 1.566, 1.74, 1.93, 2.043, 
                                       2.172, 2.322, 2.5, 2.65, 2.853, 2.964, 3.139, 3.489, 3.839, 5.191],
                                      name = "eta", label = r"$\eta$")
        
        jer_axis = hist.axis.Regular(100, 0.995, 1.030, name = 'jer', label = "JER" )
        
        
        h_njet_gen = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts")  #not in use
        h_njet_reco = hist.Hist(dataset_axis, n_axis, storage="weight", label="Counts") #not in use
        
        h_pt_reco_over_gen = hist.Hist( dataset_axis, pt_axis, frac_axis, eta_axis, pileup_axis, storage = "weight", label = "Counts")
        #h_pt_reco_over_raw = hist.Hist( dataset_axis, pt_raw_axis,n_axis, frac_axis, eta_axis, pileup_axis, storage = "weight", label = "Counts")
        
        
        h_pileup_rho = hist.Hist(dataset_axis, pileup_fine_axis, rho_fine_axis, storage = "weight", label = "Counts") #used to make pileup vs rho plot
        
        
        
        #self.df = pd.read_csv( "Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.txt", delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
        cutflow = {}

        
        self.hists = {
            "njet_gen":h_njet_gen,
            "njet_reco":h_njet_reco,
            "pt_reco_over_gen": h_pt_reco_over_gen,
            "pileup_rho": h_pileup_rho,
            "cutflow": cutflow
        }
        
    @property
    def accumulator(self):
        return self.hists
    
    def process(self, events):
        dataset = events.metadata['dataset']
        
        if dataset not in self.hists["cutflow"]:
            self.hists["cutflow"][dataset] = defaultdict(int)
            


        gen_vtx = events.GenVtx.z
        reco_vtx = events.PV.z
        
        
        # delta_z < 0.2 between reco and gen
        events = events[np.abs(gen_vtx - reco_vtx) < 0.2]
        
        
        # loose jet ID
        events.Jet = events.Jet[events.Jet.jetId > 0]
        

        events = events[ak.num(events.Jet) > 0 ]
        dataset = events.metadata['dataset']
        
        genjets = events.GenJet[:,0:3]
        #genjets = events.GenJet[:,0:6]
        recojets = genjets.nearest(events.Jet, threshold = 0.2)
        
        sel = ~ak.is_none(recojets, axis = 1)
        
        genjets = genjets[sel]
        recojets = recojets[sel]
             
        ptresponse = recojets.pt/genjets.pt
        
        n_reco_vtx = events.PV.npvs #the number of primary vertices
        n_pileup = events.Pileup.nPU #number of pileupss
        rho = events.fixedGridRhoFastjetAll
        pu_nTrueInt = events.Pileup.nTrueInt
        

        
        
        
        sel = ~ak.is_none(ptresponse,axis=1)
        ptresponse = ptresponse[sel]
        recojets = recojets[sel]
        genjets = genjets[sel]
        
        sel2 = ak.num(ptresponse) > 2
        
        recojets = recojets[sel2]
        genjets = genjets[sel2]
        
        ptresponse = ptresponse[sel2]
        ptresponse_raw = (recojets.pt * (1 - recojets.rawFactor))/genjets.pt
        
        n_reco_vtx = n_reco_vtx[sel2]
        n_pileup = n_pileup[sel2]
        rho = rho[sel2]
        pu_nTrueInt = pu_nTrueInt[sel2]
        
        n_reco_vtx = ak.broadcast_arrays(n_reco_vtx, recojets.pt)[0]
        n_pileup = ak.broadcast_arrays(n_pileup, recojets.pt)[0]
        rho = ak.broadcast_arrays(rho, recojets.pt)[0]
        pu_nTrueInt =   ak.broadcast_arrays(pu_nTrueInt, recojets.pt)[0]      
        puWeight = GetPUSF("2017", np.array(ak.flatten(pu_nTrueInt)))
        
        self.hists["pt_reco_over_gen"].fill( dataset = dataset, pt = ak.flatten(genjets.pt),frac = ak.flatten(ptresponse), 
                                            eta = np.abs(ak.flatten(genjets.eta)), pileup = ak.flatten(n_pileup), weight = puWeight)
        
        #self.hists["pt_reco_over_raw"].fill( dataset = dataset, pt_raw = ak.flatten(recojets.pt*(1 - recojets.rawFactor)), n = ak.flatten(n_reco_vtx) ,frac = ak.flatten(ptresponse_raw), eta = np.abs(ak.flatten(genjets.eta)), pileup = ak.flatten(n_pileup))
        
        #self.hists["pileup_rho"].fill(dataset = dataset, rho_fine = ak.flatten(rho), pileup_fine = ak.flatten(n_pileup), weight = puWeight)
            
        return self.hists
    
    def postprocess(self, accumulator):
        return accumulator
        