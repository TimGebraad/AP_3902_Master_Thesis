# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:40:46 2016

@author: timge
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import math

class NullWriter(object):
    def write(self, arg):
        pass

def DisplayMatrix (Matrix):
    fig, ax = plt.subplots(figsize=(20,11))
    table = plt.table(cellText = np.round(Matrix, 8), cellLoc='center', bbox=[0, 0, 1, 1])
    table.set_fontsize(25)
    ax.axis('off')
    plt.show()
    
def DisplayMatrixColor (Matrix):
    fig, ax = plt.subplots(figsize=(20,11))
    ax.imshow(Matrix, interpolation='none')
    plt.show()
    
def make_colormap(seq):
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def LogLinLogSpace(bnd = [-1000, -10, 10, 100], N=1000, bPlot=False):
    dx0 = (bnd[2]-bnd[1])/N
    
    lGrid, cGrid, rGrid = [[], [], []]
    
    if bnd[0]<bnd[1]:
        lGrid = bnd[1]-np.logspace(start=np.log(dx0), stop=np.log(bnd[1]-bnd[0]), num=abs(10*math.ceil((np.log(bnd[1]-bnd[0])-np.log(dx0))/np.log(2*dx0))), base=np.exp(1))[::-1]
        #Replace points that are spaced with less than dx0 with linear spacing
#        try:
#            nMin = np.min(np.argwhere((lGrid[1:]-lGrid[0:-1])<dx0))
#            lGrid = np.concatenate((lGrid[0:nMin-1], np.linspace(lGrid[nMin], bnd[1]-dx0, int((bnd[1]-lGrid[nMin])/dx0))))
#        except: pass
    if bnd[1]!=bnd[2]:
        cGrid = np.linspace(bnd[1], bnd[2], N)
    if bnd[2]<bnd[3]:
        rGrid = bnd[2]+np.logspace(start=np.log(dx0), stop=np.log(bnd[3]-bnd[2]), num=abs(10*math.ceil((np.log(bnd[3]-bnd[2])-np.log(dx0))/np.log(2*dx0))), base=np.exp(1))
#        try:
#            nMax = np.max(np.argwhere((rGrid[1:]-rGrid[0:-1])<dx0))
#            rGrid = np.concatenate((np.linspace(bnd[2]+dx0, rGrid[nMax], int((rGrid[nMax]-bnd[2])/dx0)), rGrid[nMax+1:]))
#        except: pass
    x = np.concatenate((lGrid, cGrid, rGrid))    

    if bPlot:  
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, np.zeros(len(x)), 'kx')
        ax.plot((x[1:]+x[0:-1])/2, x[1:]-x[0:-1], 'r.')
    return x