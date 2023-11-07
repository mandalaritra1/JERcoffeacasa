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

#Function to read the JRfiles and determine the JER for certain ranges of eta, rho and pt

def computeJER(pt, eta, rho, filename):
    df = pd.read_csv( filename, delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
    
    df = df[ (eta > df['eta_low']) &  (eta <= df['eta_high']) & (rho > df['rho_low']) & (rho <= df['rho_high'])  ]
    p0 = df['par0']
    p1 = df['par1']
    p2 = df['par2']
    p3 = df['par3']
    x = pt
    return np.sqrt(p0*np.abs(p0)/(x*x)+p1*p1*np.power(x,p3) + p2*p2)


def GetPUSF(IOV, nTrueInt, var='nominal'):
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
    fname = "data/puWeights"+IOV+"UL.json.gz"
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)
    return evaluator[hname[IOV]].evaluate(np.array(nTrueInt), var)