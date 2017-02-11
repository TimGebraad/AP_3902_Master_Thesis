# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:19:32 2017

@author: timge
"""

#Import general modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
from numpy import inf

#Import custom modules
from Molecule import *
from Graphics import *

class Magnetics(object):
    def __init__(self, Mol):
        self.Mol = Mol
        
    def GetMagneticField(self, T, R):
        x = self.Mol.Atom[:,1]
        y = self.Mol.Atom[:,2]

        N = 3
        
        xx2, xx1 = np.meshgrid(x, x)
        yy2, yy1 = np.meshgrid(y, y)
        
        dx = xx2-xx1
        dy = yy2-yy1
        
        dl = np.sqrt((dx)**2+(dy)**2)/N

        dlcrossrhat = np.zeros((self.Mol.N, self.Mol.N))
        
        for l in range(N):
            X1 = xx1+dx/N*l
            X2 = xx1+dx/N*(l+1)
            
            Y1 = yy1+dy/N*l
            Y2 = yy1+dy/N*(l+1)
                             
            Cx = (X1+X2)/2
            Cy = (Y1+Y2)/2
        
            R2 = (Cx-R[0])**2+(Cy-R[1])**2+R[2]**2        
            R_3 = 1/np.sqrt(R2)**3
            R_3[R_3 == inf] = 0
        
            dlcrossrhat += ((xx2-xx1)*(Cy-R[1])-(yy2-yy1)*(Cx-R[0]))*R_3/dL*dl

        B = np.sum(T*dlcrossrhat)/2
        B *=10**(-7)
        
        return B
        
    def GetMagneticFieldArea(self, X, Y, z, I, ax=None):        
        B = np.zeros((len(X), len(Y)))
        
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                B[i,j] = self.GetMagneticField(I, [x, y, z],)
        
        if ax!=None:
            YY, XX = np.meshgrid(Y,X)
            linthres = 10**(-15)    #abs(B).mean()
            vmax     = 10**(-8)     #abs(B).max()
            norm = colors.SymLogNorm(linthresh=linthres, vmin=-vmax, vmax=vmax)
            
            im = ax[1].pcolormesh(XX, YY, B,  cmap = 'RdBu_r', norm=norm)
            ax[1].scatter(self.Mol.Atom[:,1], self.Mol.Atom[:,2], color='k')
            tick_locations=([-(10**x) for x in range(-15, 8, 1)][::-1]
                    +[0.0]
                    +[(10**x) for x in range(-15,8, 1)] )
            cb = ax[0].colorbar(im, ax=ax[1], extend='both', label='$Magnetic field$', ticks = tick_locations, format=ticker.LogFormatterMathtext())
            
            
#            cb.colorbar(ticks=tick_locations, format=ticker.LogFormatterMathtext())
        
        return B
        
    def DrawLocalTransmission(self, I, ax):
        ax.scatter(self.Mol.Atom[:,1], self.Mol.Atom[:,2], color='k')
        LCPlot = np.ndarray((2, 2, self.Mol.N, self.Mol.N), dtype=np.object)
        for i in range(self.Mol.N):
            for j in range(self.Mol.N):
                if i!=j and self.Mol.Ham[i,j]!=0:
                    x0 = self.Mol.Atom[i][1]
                    y0 = self.Mol.Atom[i][2]
                    dx = self.Mol.Atom[j][1]-x0
                    dy = self.Mol.Atom[j][2]-y0
                    for spin in range(2):
                        for lead in range(1):        
                            #Summed by leads                     
                            LCPlot[spin, lead, i,j] = ax.arrow(x0+0.1*dx, y0+0.1*dy, dx*0.8, dy*0.8, \
                            head_width=0.20, head_length=+0.1, lw=0, fc='r', ec='y', \
                            shape=('left' if spin else 'right'), \
                            alpha= 0.5, visible=True)
                                
        maxCurrent = abs(I).max()
        for spin in range(2):
            for lead in range(1):
                for i in range(self.Mol.N):
                    for j in range(i+1,self.Mol.N):
                        Inet = (I[spin, lead, j,i]-I[spin, lead, i,j])/2
                        color = 'limegreen'
                        if Inet>0:
                            LCPlot[spin, lead, i,j].set_linewidth(Inet/maxCurrent*10)
                            LCPlot[spin, lead, i,j].set_color(color)
                            LCPlot[spin, lead, i,j].set_visible(True)
                            LCPlot[spin, lead, j,i].set_visible(False)
                        elif Inet<0:
                            LCPlot[spin, lead, j,i].set_linewidth(-Inet/maxCurrent*10)
                            LCPlot[spin, lead, j,i].set_color(color)
                            LCPlot[spin, lead, j,i].set_visible(True)
                            LCPlot[spin, lead, i,j].set_visible(False)
                        elif i!=j and self.Mol.Ham[i,j]!=0:
                            LCPlot[spin, lead, i,j].set_visible(False)
                            LCPlot[spin, lead, j,i].set_visible(False)
        
    def PerformAnalysis(self):
        E = np.linspace(7, 9, 21)
        self.Mol.SetBias(E[1]-E[0])
        
        try:
            X = np.linspace(-dL, self.Molasdf.Rc[0]+dL, 100)
            Y = np.linspace(-dL, self.Mol.Rc[1]+dL, 100)
        except:
            X = np.linspace(-dL, np.max(self.Mol.Atom[:,1])+dL, 100)
            Y = np.linspace(-dL, np.max(self.Mol.Atom[:,2])+dL, 100)
            
        
#        X = np.linspace(-dL, 13, 40)
#        Y = np.linspace(-dL, 13, 40)
            
        Itot = np.sum(self.Mol.CurrentAnalytical(E-self.Mol.Bias/2+self.Mol.Efermi, E+self.Mol.Bias/2+self.Mol.Efermi), axis=0)
        x, y = np.zeros((2,len(E))) 
        Bmax = np.zeros(len(E))
        ratio = np.zeros(len(E))
        
        for i, e in enumerate(E):
            I = self.Mol.LocalCurrent(e+self.Mol.Efermi)
            I /= 10**9
            
            fig = plt.figure()
            ax1, ax2 = [fig.add_subplot(211), fig.add_subplot(212)]

            self.DrawLocalTransmission(I, ax1)
            B = self.GetMagneticFieldArea(X, Y, 1, np.sum(np.sum(I, axis=1), axis=0), [fig, ax2])
            
            Bmax[i] = np.max(abs(B))
            ratio[i] = Bmax[i]/Itot[i]*10**9
            xi, yi = np.unravel_index(abs(B).argmax(), abs(B).shape)
            x[i], y[i] = X[xi], Y[yi] 
            
            ax1.axis('equal')
            ax1.set_xlim(np.min(X), np.max(X))
            ax1.set_ylim(np.min(Y), np.max(Y))
            ax1.set_xlabel('$x \ [\AA{}]$')
            ax1.set_ylabel('$y \ [\AA{}]$')
            ax2.axis('equal')
            ax2.set_xlim(np.min(X), np.max(X))
            ax2.set_ylim(np.min(Y), np.max(Y))
            ax2.set_xlabel('$x \ [\AA{}]$')
            ax2.set_ylabel('$y \ [\AA{}]$')
            
            fig.set_size_inches(10.5, 20.5)
            fig.savefig(self.Mol.szOutputFolder + self.Mol.Name + ' - ' + str(i) + " - E = " + str(int(100*e)/100) + ".png",  dpi=fig.dpi)
        
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.semilogy(E, Itot, 'darkblue')
        ax.set_ylabel('$Current \ [nA]$')
        ax = ax.twinx()
        ax.semilogy(E, Bmax, 'darkred')
        ax.set_ylabel('$Magnetic field \ [T]$')
        ax = fig.add_subplot(413)
        print(E)
        print(ratio)
        print(Itot)
        ax.plot(E, ratio, 'k:')
        ax.set_ylabel('$Ratio$')
        ax = fig.add_subplot(414)
        ax.plot(E, x, 'lightblue')
        ax.plot(E, y, 'lightcoral')
        fig.savefig(self.Mol.szOutputFolder + self.Mol.Name + " - Magnetic field analysis.png",  dpi=fig.dpi)
        
        return Bmax
        
        
        
if __name__ == '__main__':
    N = 3
    L = 4*int(np.sqrt(3)/3*N)
    L += L%2
    
    Mol = AA_Corner_GNR(60, N, N, L, L)
#    Mol = AGNR(3,2)
#    Mol.SetLead(0, 'Left')
#    Mol.SetLead(3, 'Right')
#    Mol.SetLeads()
    Mol.SetGam_b(0.0000001)
    Mol.SetBias(0.1)
    Mol.UpdateMolecule()
    
    Magnetics = Magnetics(Mol)
    Magnetics.PerformAnalysis()
    
#    Graph = Graphics()
#    Graph.SetMolecule(Mol)
    
    
    plt.show()