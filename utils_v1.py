import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
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


def returnParm(eta, rho, filename):
    df = pd.read_csv( filename, delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
    
    df = df[ (eta > df['eta_low']) &  (eta <= df['eta_high']) & (rho > df['rho_low']) & (rho <= df['rho_high'])  ]
    p0 = df['par0']
    p1 = df['par1']
    p2 = df['par2']
    p3 = df['par3']
    return np.reshape([p0, p1, p2, p3], 4)

def computeJER(pt, eta, rho, filename):
    df = pd.read_csv( filename, delimiter='\s+', skiprows = 1, names = ['eta_low','eta_high', 'rho_low', 'rho_high', 'unknown','pt_low','pt_high','par0','par1','par2','par3'])
    
    df = df[ (eta > df['eta_low']) &  (eta <= df['eta_high']) & (rho > df['rho_low']) & (rho <= df['rho_high'])  ]
    p0 = df['par0']
    p1 = df['par1']
    p2 = df['par2']
    p3 = df['par3']
    x = pt
    return np.sqrt(p0*np.abs(p0)/(x*x)+p1*p1*np.power(x,p3) + p2*p2)

def mean_finder(hist, centers):
    sum = 0
    weight_sum = 0
    
    for i in range(len(hist)):
        sum = sum + hist[i]*centers[i]
        weight_sum = weight_sum + hist[i]
    return sum/weight_sum

def std_finder(hist, centers):
    sum = 0
    weight_sum = 0
    mean = mean_finder(hist, centers)
    
    for i in range(len(hist)):
        sum = sum + hist[i]*((centers[i] - mean)**2)
        weight_sum = weight_sum + hist[i]
        
    return np.sqrt(sum/(weight_sum - 1))

def sem_finder(hist, centers): #finds standard error of mean
    return std_finder(hist,centers)/(np.sum(hist)**0.5)


def std_finder(hist, centers):
    sum = 0
    weight_sum = 0
    mean = mean_finder(hist, centers)
    
    for i in range(len(hist)):
        sum = sum + hist[i]*((centers[i] - mean)**2)
        weight_sum = weight_sum + hist[i]
        
    return np.sqrt(sum/(weight_sum - 1))


# def reduce_bin(hist_values, bin_widths, bin_edges, bin_centers, factor = 2):
#     hist_values = sum_adjacent_pairs(hist_values)
#     bin_edges = np.array(bin_edges)[::2]
#     bin_widths = np.diff(bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#     return hist_values, bin_centers

def reduce_bins( counts, bin_edges, n):
    new_bin_edges = bin_edges[::n]
    new_counts = np.zeros(len(new_bin_edges) - 1)
    for i in range(len(new_counts)):
        new_counts[i] = np.sum(counts[i*n:(i+1)*n])
    new_bin_centers = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
    return  new_counts   , new_bin_centers

def sum_adjacent_pairs(arr):
    arr = np.array(arr)
    
    # Pad the array with a zero if the length is odd
    if len(arr) % 2 != 0:
        arr = np.pad(arr, (0, 1), 'constant')

    # Reshape the array and sum along the second axis
    result = arr.reshape(-1, 2).sum(axis=1)

    return result



class Histfit:
    def __init__(self, hist_frac_pt, frac_axis, pt_values, variable = 'pt'):
        self.frac_values = frac_axis.centers
        self.frac_edges = frac_axis.edges
        self.frac_widths = frac_axis.widths
        self.hist_frac_pt = hist_frac_pt
        self.pt_values = pt_values
        
        
        self.parameters = {"mean":np.full(len(self.pt_values), None), "sigma": np.full(len(self.pt_values), None), "meanErr":np.full(len(self.pt_values), None),"sigmaErr":np.full(len(self.pt_values), None)}
        
    def gauss(self,x,  x0, sigma,a):
        return (a*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)))
    
    # def gauss_2(self,x,  x0, sigma):
    #     return ((1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)))
    
    def fitGauss(self, hist_frac, frac_values):
        mean = mean_finder(hist_frac, frac_values)
        std = std_finder(hist_frac, frac_values)
        max_value = np.max(hist_frac) 
        
        parameters, covariance = curve_fit(self.gauss, frac_values, hist_frac,p0 = [mean, std, max_value], bounds = ([0.5,0.001,0],[2,1.0,2000000]) ) 
        mean = parameters[0]
        sigma = parameters[1]
        const = parameters[2]
        meanErr = covariance[0][0]
        sigmaErr = covariance[1][1]
        return mean,sigma,meanErr, sigmaErr, const
    
   
    
    def initiate_parameters(self):
        
        for i in range(len(self.hist_frac_pt)):
            hist_frac = self.hist_frac_pt[i]
            
            if np.sum(hist_frac) > 300:
                results = self.fitGauss(hist_frac, self.frac_values)
                for j,key in enumerate(self.parameters.keys()):
                    self.parameters[key][i] = results[j]
            else:
                for j,key in enumerate(self.parameters.keys()):
                    self.parameters[key][i] = None
                break
    
    def store_parameters_ci(self):
        for i in range(len(self.hist_frac_pt)):
            hist_frac = self.hist_frac_pt[i]
        
            if np.sum(hist_frac) > 500:
                confLevel = 0.875
                ix =  np.argmax(hist_frac)
                ixlow = ix
                ixhigh = ix
                nb = np.shape(hist_frac)
                ntot = np.sum(hist_frac)
                nsum = hist_frac[ix]
                width = self.frac_values[2] - self.frac_values[1]
                nlow = 0
                nhigh = 0

                while (nsum < confLevel*ntot) :
                    if (ixlow>0):
                        nlow = hist_frac[ixlow -1]
                    else:
                        nlow = 0

                    if (ixhigh < nb):
                        nhigh = hist_frac[ixhigh +1]
                    else:
                        nhigh = 0

                    if (nsum + np.max([nlow,nhigh]) < confLevel*ntot):
                        if( nlow >= nhigh and ixlow > 0):
                            nsum = nsum + nlow
                            ixlow = ixlow -1
                            width = width + width
                        elif (ixhigh < nb):
                            nsum = nsum + nhigh
                            ixhigh = ixhigh + 1
                            width = width + width
                    else:
                        if (nlow > nhigh):
                            width = width + width*(confLevel*ntot - nsum)/nlow
                        else:
                            width = width + width*(confLevel*ntot - nsum)/nhigh
                        nsum = ntot
                self.cipar.append(width/(1.514*2))
            else:
                break
                
            
            
    def store_parameters(self):
        self.initiate_parameters()
        for repeater in range(10):
            for i,hist_frac in enumerate(self.hist_frac_pt):
                frac_values = self.frac_values
        
                
            
                if np.sum(hist_frac) > 300:
                    if np.sum(hist_frac)<10000:
                        hist_frac_full, frac_values_full = reduce_bins(hist_frac, self.frac_edges, 4)
                    if (np.sum(hist_frac) >= 10000) and (np.sum(hist_frac)< 20000):
                        hist_frac_full, frac_values_full = reduce_bins(hist_frac, self.frac_edges, 2)
                    if (np.sum(hist_frac) >= 20000):
                        hist_frac_full, frac_values_full = reduce_bins(hist_frac, self.frac_edges, 1)
                    hist_frac = hist_frac_full
                    frac_values = frac_values_full
                    sel = (frac_values > (self.parameters["mean"][i] - 2*self.parameters["sigma"][i])) &  (frac_values < (self.parameters["mean"][i] + 2*self.parameters["sigma"][i]))
                    frac_values = frac_values[sel]
                    hist_frac = hist_frac[sel]

                    results = self.fitGauss(hist_frac, frac_values)

                    if np.abs(results[2] - self.parameters["sigma"][i] ) < 0.000001:
                        break
                    for j,key in enumerate(self.parameters.keys()):
                        self.parameters[key][i] = results[j]
                else:
                    break
    def show_fit(self, i):
        hist_frac = self.hist_frac_pt[i]
        frac_values = self.frac_values
        
        hist_frac_full, frac_values_full = reduce_bins(hist_frac, self.frac_edges, 1)
        hist_frac = hist_frac_full
        frac_values = frac_values_full
        print(f"Showing plot for pt = {self.pt_values[i]}")
        print(f"length of reduced array {len(hist_frac)}")
        print("Total number of events in this bin " + str(np.sum(hist_frac[1:])))
        print(self.parameters["mean"][i])
        print(self.parameters["sigma"][i])
        sel = (frac_values > (self.parameters["mean"][i] - 1.5*self.parameters["sigma"][i])) &  (frac_values < (self.parameters["mean"][i] + 1.5*self.parameters["sigma"][i]))
        
        #print(sel)
        #print(hist_frac)
        
        frac_values = frac_values[sel]
        hist_frac = hist_frac[sel]
        
        results = self.fitGauss(hist_frac[1:], frac_values[1:])
        
        print("Mean: {} ".format(results[0]))
        print("Width: {}".format(results[1]))
        for j,key in enumerate(self.parameters.keys()):
                    self.parameters[key][i] = results[j]
                
        #plt.plot(self.frac_values, self.hist_frac_pt[i], 'b-', label = "Response")
        plt.plot(frac_values_full, hist_frac_full, 'b-', label = "Response")
        plt.plot(frac_values, self.gauss(frac_values, results[0], results[1], results[4]), 'black',linestyle = '--' ,label = "Gauss Fit")
        

            
