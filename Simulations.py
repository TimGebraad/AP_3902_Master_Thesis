# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:12:26 2016

@author: timge
"""

#Import general modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#Import custom modules
from Lifting import *
from Graphics import *
from ImportExport import *
from Miscellaneous import *

def Simulate(Mol, Simulations):
        
    if 'Graphics' in Simulations:
        #Save a graph of the molecule and the 4 states around the Fermi Energy
        Graph = Graphics()
        Graph.SetMolecule(Mol)
        Graph.Save(Mol.szOutputFolder + Mol.Name + " - Molecule.png", part='Molecule')
        Graph.UpdateAtomSel(np.array([np.real(Mol.eigvec[0][:,Mol.N/2-2]*np.conjugate(Mol.eigvec[0][:,Mol.N/2-2])), np.zeros(Mol.N)]))
        Graph.Save(Mol.szOutputFolder + Mol.Name + " - Orbital - HOMO 2.png", part='Molecule')
        Graph.UpdateAtomSel(np.array([np.real(Mol.eigvec[0][:,Mol.N/2-1]*np.conjugate(Mol.eigvec[0][:,Mol.N/2-1])), np.zeros(Mol.N)]))
        Graph.Save(Mol.szOutputFolder + Mol.Name + " - Orbital - HOMO 1.png", part='Molecule')
        Graph.UpdateAtomSel(np.array([np.real(Mol.eigvec[0][:,Mol.N/2-0]*np.conjugate(Mol.eigvec[0][:,Mol.N/2-0])), np.zeros(Mol.N)]))
        Graph.Save(Mol.szOutputFolder + Mol.Name + " - Orbital - LUMO 1.png", part='Molecule')
        Graph.UpdateAtomSel(np.array([np.real(Mol.eigvec[0][:,Mol.N/2+1]*np.conjugate(Mol.eigvec[0][:,Mol.N/2+1])), np.zeros(Mol.N)]))
        Graph.Save(Mol.szOutputFolder + Mol.Name + " - Orbital - LUMO 2.png", part='Molecule')
        
        
    if 'Lifting' in Simulations:
        #Set up the plots
        fig = plt.figure()
        axLift, axOrbs = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
        
        #Load and plot the experimental data if available
        for root, dirs, files in os.walk(Mol.szOutputFolder + 'Experimental Data/'):
            for file in files:        
                print(file)                          
                header, values = ImportData(Mol.szOutputFolder + 'Experimental Data/' + file)
                axLift.plot(10*values[:,0], values[:,1], 'k', marker='.', lw=0, label=file, alpha=0.1)

                
        #Create MD folder if it did not already exists
        if not os.path.exists(Mol.szOutputFolder + 'MD'):
            os.makedirs(Mol.szOutputFolder + 'MD')            
                
        #Perform lifting simulation
        MD = MD_Lifting(Mol)
        dz = np.linspace(0,np.max(Mol.Atom[:,1])-0.1, 50)
#        Mol.SetBias(0.01)
        I = MD.PerformLifting(dz, export=['param', 'dzI', 'xyz'], axOrb=axOrbs)
                
        #Plot tweaking and saving
        axLift.plot(dz, np.log(I), color='midnightblue', label='Simulated', lw=2, alpha=0.8)
        axLift.plot(dz, np.log(I), color='midnightblue', label='Simulated', lw=0, alpha=0.8, marker='o', markersize=10)
#        axLift.legend()
#        axLift.set_title('Lifting')
        axLift.set_xlabel('$\Delta \ z [A]$')
        axLift.set_ylabel('$ln(I)$')                
        axLift.set_xlim([np.min(dz), np.max(dz)])
        ymin, ymax = np.min(np.log(I)), np.max(np.log(I))
        axLift.set_ylim([ymin-0.1*(ymax-ymin), ymax+0.1*(ymax-ymin)])
        
#        #Plot Beta factor
#        axBeta = axLift.twinx()
#        axBeta.plot((dz[0:-4]+dz[4:])/2, (np.log(I[0:-4])-np.log(I[4:]))/(dz[4]-dz[0])*10, c='r', alpha=0.5, lw=2)
#        axBeta.set_ylabel('$Beta$')


        axOrbs.set_title('Orbitals')
        axOrbs.set_xlabel('$\Delta z [A]$')
        axOrbs.set_ylabel('$E_{Fermi} [eV]$')
        axOrbs.legend(['HOMO 2', 'HOMO 1', 'LUMO 1', 'LUMO 2'])
            
        fig.set_size_inches(20.5, 6.5)
        mpl.rcParams.update({'font.size': 20})
        fig.savefig(Mol.szOutputFolder +  Mol.Name + ' - Lifting vs Current.png',  dpi=fig.dpi)
        
        
    if 'LiftingSpectroscopy' in Simulations:
        #Set up parameters and perform simulation
        MD = MD_Lifting(Mol)
        dz = np.linspace(0,np.max(Mol.Atom[:,1])-0.1, 50)
        neg_bias = -np.linspace(0.015, 0.5, 100)[::-1]
        pos_bias =  np.linspace(0.005, 0.5, 100)
        I = MD.PerformLiftingSpectroscopy(dz, np.concatenate((neg_bias, pos_bias)), export=['dz-V-I'])
        I = [I[:, 0:len(neg_bias)], I[:, len(neg_bias):]]
        
        #Set up the figure
        c = mcolors.ColorConverter().to_rgb
        fig = plt.figure()
        axI, axdIdV, axBeta = [fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1,3,3)]
              
        #Plot ln(I) data
        BIAS, DZ = [[None, None], [None, None]]
        for i, bias in enumerate([neg_bias, pos_bias]):
            BIAS[i], DZ[i] = np.meshgrid(bias, dz)
        im = axI.contourf(np.concatenate([DZ[0], DZ[1]], axis=1), np.concatenate([BIAS[0], BIAS[1]], axis=1), np.log(abs(np.concatenate([I[0], I[1]], axis=1))), 100, cmap = make_colormap([c('white'), c('midnightblue')]))
        axI.set_ylabel('$Bias$ [$V$]')
        axI.set_xlabel('$\Delta z \ [\AA{}]$')
        fig.colorbar(im, ax=axI, label='$ln(I)$')
        
        #Plot dI/dV data
        dIdV, mBIAS, mDZ = [[None, None],[None, None],[None, None]]
        for i, bias in enumerate([neg_bias, pos_bias]):
            dIdV[i] = (I[i][:,1:]-I[i][:,0:-1])/(bias[1]-bias[0])
            mBIAS[i], mDZ[i] = np.meshgrid(0.5*(bias[1:]+bias[0:-1]), dz)
        im = axdIdV.contourf(np.concatenate([mDZ[0], mDZ[1]], axis=1), np.concatenate([mBIAS[0], mBIAS[1]], axis=1), np.log(np.concatenate([dIdV[0], dIdV[1]], axis=1)), 100, cmap = make_colormap([c('white'), c('darkgreen')]))
        axdIdV.set_ylabel('$Bias$ [$V$]')
        axdIdV.set_xlabel('$\Delta z \  [\AA{}]$')
        fig.colorbar(im, ax=axdIdV, label='$ln(dI/dV)$')
        
        #Plot beta factor
        Beta, BIAS, DZ = [[None, None],[None, None],[None, None]]
        for i, bias in enumerate([neg_bias, pos_bias]):
            Beta[i] = (np.log(abs(I[i][1:,:]))-np.log(abs(I[i][0:-1,:])))/(dz[1]-dz[0])
            BIAS[i], DZ[i] = np.meshgrid(bias, 0.5*(dz[1:]+dz[0:-1]))
        im = axBeta.contourf( np.concatenate([DZ[0], DZ[1]], axis=1), np.concatenate([BIAS[0], BIAS[1]], axis=1),np.concatenate([Beta[0], Beta[1]], axis=1), 100, cmap = make_colormap([c('darkred'), c('white')]))
        axBeta.set_ylabel('$Bias$ [$V$]')
        axBeta.set_xlabel('$\Delta \ z$ [$\AA{}]$')
        fig.colorbar(im, ax=axBeta, label='$Beta factor$')
        mpl.rcParams.update({'font.size': 42})
        fig.set_size_inches(100.5, 13.5)
        fig.savefig(Mol.szOutputFolder +  Mol.Name + ' - LiftingSpectroscopy.png',  dpi=fig.dpi)
        
        
    if 'MD' in Simulations:
        #Perform a molecular dynamics simulation only
        MD = MD_Lifting(Mol)
        MD.PerformLiftingMD(np.linspace(0,np.max(Mol1.Atom[:,1])-0.1, 50), export=['xyz'])