#Import general modules
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.widgets import CheckButtons, RadioButtons, Button, Slider
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math as math
import matplotlib.cm as cm
from numpy.random import randn
import datetime
import sys

#Import custom modules
from AtomParam import *
from Miscellaneous import *


class GraphMolecule (object):
    """ Manages the displaying of the molecule and selected atoms, currents, densities
        Attributes
        axMol                   Handle to the window with the atoms
        axAtoms                 Handle to the individual atoms
        AtomPlot                Handle to the atom indicator spot
        AtomSel                 Index of the selected atom
        LDPlot                  Handle to the local densities
        LCPlot                  Handle to the local current arrows

        Methods
        InitMolecule()
        ChangeMolecule()        Creates handles to the density and selection data points
        DrawMolecule()          Draws the molecule itself
        DrawLocalDensity(bDraw) Draws the local density onto the atom sites
        DrawLocalCurrents(bDraw)Draws the local currents from the atoms to each other
        OnClickMolecule(event)  Unselect atom for double click
        OnPickMolecule(event)   Select clicked atom
        OnPress()               Select next/previous atom or change atoms 
        DrawSelAtom()           Draws the indicator for the selected atom or series of atoms
        """
    
    def InitMolecule (self):
        start = datetime.datetime.now()
        self.axMol = self.axBias.twinx()
        self.axBias.yaxis.set_ticks_position('none') 
#        self.axBias.set_axis_bgcolor('black')
        self.axMol.set_position (self.axBias.get_position())
        self.AtomSel = None
        self.LCPlot = None
        
        #Set up plots for Atoms, Local Density (LD) and Local Density of States (LDOS)
        self.scatAtoms = self.axMol.scatter([], [])
        self.scatLD    = self.axMol.quiver ([], [], [], [])
        self.scatLDOS = [self.axMol.scatter([], []), self.axMol.scatter([], [])]
                
        print('   $GraphMolecule - Init:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

        
    def ChangeMolecule(self):
        start = datetime.datetime.now()

        if self.LCPlot is not None:
            for spin in range(2):
                for lead in range(2):
                    for i in range(self.Mol.N):
                        for j in range(self.Mol.N):
                            if i!=j and self.Mol.Ham[i,j]!=0:
                                self.LCPlot[spin, lead, i,j].remove()
        del self.LCPlot
        
        self.LCPlot = np.ndarray((2, 2, self.Mol.N, self.Mol.N), dtype=np.object)
        for i in range(self.Mol.N):
            for j in range(self.Mol.N):
                if i!=j and self.Mol.Ham[i,j]!=0:
                    x0 = self.Mol.Atom[i][1]
                    y0 = self.Mol.Atom[i][2]
                    dx = self.Mol.Atom[j][1]-x0
                    dy = self.Mol.Atom[j][2]-y0
                    for spin in range(2):
                        for lead in range(1+self.bSplitLeads):         
                            if not self.bSplitLeads:
                                #Summed by leads                     
                                self.LCPlot[spin, lead, i,j] = self.axMol.arrow(x0+0.1*dx, y0+0.1*dy, dx*0.8, dy*0.8, \
                                head_width=0.20, head_length=+0.1, lw=0, fc='r', ec='y', \
                                shape=('left' if spin else 'right'), \
                                alpha= 0.5, visible=False)
                            else:
                                #Seperated into leads
                                bLeft = (self.Mol.Atom[i][1]<=self.Mol.Atom[j][1] and lead) or (self.Mol.Atom[i][1]>self.Mol.Atom[j][1] and not lead)
                                self.LCPlot[spin, lead, i,j] = self.axMol.arrow(x0+(0.1+0.5*bLeft)*dx, y0+(0.1+0.5*bLeft)*dy, dx*0.3, dy*0.3, \
                                head_width=0.20, head_length=+0.1, lw=0, fc='r', ec='y', \
                                shape=('left' if spin else 'right'), \
                                alpha= (1.0 if bLeft else 0.5), ls= (':' if lead else '-'), visible=False)

        print('   $GraphMolecule - ChangeMolecule:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawMolecule (self):
        start = datetime.datetime.now()
        x = self.Mol.Atom[:,1]
        y = self.Mol.Atom[:,2]
        
        #Draw nearest neighbour raster
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                if abs(self.Mol.Ham[i,j])>1.0:
                    self.axMol.plot ([self.Mol.Atom[i,1], self.Mol.Atom[j,1]], [self.Mol.Atom[i,2], self.Mol.Atom[j,2]], linewidth=1, color='k', alpha=0.8)        

        #Draw lead connections
        self.axMol.plot(x[self.Mol.lpL]-0.15, y[self.Mol.lpL], '>y', markersize=7, alpha=0.5)
        self.axMol.plot(x[self.Mol.lpR]+0.15, y[self.Mol.lpR], '<y', markersize=7, alpha=0.5)
        self.axMol.scatter(x, y-0.15, c='y', marker='^', s=np.sqrt(np.real(self.Mol.lpS)/np.real(Gam_S))*70, lw=0, alpha=0.5)
        
        #Draw the atoms
        self.scatAtoms.remove()
        self.scatAtoms = self.axMol.scatter(x,y, s=60, c=[cAtom[AtomNr[i]] for i in self.Mol.Atom[:,0]], lw=0, picker=True, zorder=2)

        #Plot tweaking
        self.axMol.set_xlabel('$x \ [\AA{}]$')
        self.axMol.xaxis.set_label_position('top')
        self.axBias.xaxis.tick_top()
        self.axMol.set_ylabel('$y \ [\AA{}]$')

        self.axMol.axis('equal')
        self.axMol.relim()
        self.axMol.autoscale_view()
#        self.axBias.set_xlim([-5, 30])
        
        print('   $GraphMolecule - DrawMolecule:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def DrawLocalDensity(self, bDraw=None):
        start = datetime.datetime.now()
        self.scatLD.remove()
        
        if bDraw==None:
            bDraw = self.cbOptions1.lines[0][0].get_visible()
            
        if bDraw:
            x = self.Mol.Atom[:,1]
            y = self.Mol.Atom[:,2]
            color = 'greenyellow'
            LD = np.concatenate((self.Mol.LD[0], self.Mol.LD[1]))
            x  = np.concatenate((x+0.1, x-0.1))
            y  = np.concatenate((y, y))
            dx = np.zeros(self.Mol.N*2)
            dy = np.concatenate((np.ones(self.Mol.N), -np.ones(self.Mol.N)))*0.2
            color = LD
            N = (self.Mol.bSpin+1)*self.Mol.N
            size = abs(LD)/np.mean(LD[0:N])
            
            #Set colorbar
            norm = mpl.colors.Normalize(np.min(LD), np.max(LD))
            self.cbLD = mpl.colorbar.ColorbarBase(self.axLDcb, cmap=self.cmapLD, norm=norm, orientation='vertical')
            self.cbLD.set_label('DOS')  
            
            self.scatLD = self.axMol.quiver(x[0:N], y[0:N], dx[0:N], dy[0:N]*size[0:N], color[0:N], alpha=0.3, cmap=self.cmapLD, zorder=2)
        else:
            self.scatLD = self.axMol.quiver([], [], [], [])
            
        print('   $GraphMolecule - DrawLocalDensity:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawPotential(self):
        return #temporary
        start = datetime.datetime.now()    
        norm = mpl.colors.Normalize(np.min(self.Mol.PotExt), np.max(self.Mol.PotExt))
        self.cbPot = mpl.colorbar.ColorbarBase(self.axPotcb, cmap=self.cmapPot, norm=norm, orientation='horizontal')
        self.cbPot.set_label('$Potential [eV]$')
#        plt.setp(self.cbPot.ax.xaxis.get_ticklabels(), color='azure')

        self.axMol.pcolormesh(self.Mol.X, self.Mol.Y, self.Mol.PotExt, cmap=self.cmapPot, zorder=1)
        print('   $GraphMolecule - DrawPotential:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawLDOS (self):
        return #temporary
        start = datetime.datetime.now()
        for spin in range(2):
            try:
                self.scatLDOS[spin].remove()
            except:
                pass
            
        if isinstance(self.AtomSel, np.ndarray):
            for spin in reversed(range(self.Mol.bSpin+1)):
                size = self.AtomSel[spin]/np.mean(self.AtomSel[spin])*250
                color= self.AtomSel[spin]*self.Mol.N
                marker = 'o' if not self.Mol.bSpin else list(zip(np.cos(np.radians(np.linspace(0, 180, 10))), np.sin(np.radians(np.linspace(0, 180, 10))*(1-2*spin))))             
                self.scatLDOS[spin] = self.axMol.scatter(self.Mol.Atom[:,1], self.Mol.Atom[:,2], s=size, c=color, lw=0, cmap=self.cmapLDOS, vmin=0, vmax=self.axDOS.get_ylim()[1], alpha=0.4, marker=marker)
        else:
            for spin in range(self.Mol.bSpin+1):
                self.scatLDOS[spin] = self.axMol.scatter([], [])
            if self.AtomSel!=None:
                    self.scatLDOS[0] = self.axMol.scatter(self.Mol.Atom[self.AtomSel][1], self.Mol.Atom[self.AtomSel][2], s=100, c='r')
        print('   $GraphMolecule - DrawLDOS:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawLocalCurrents(self, bDraw=None):
        start = datetime.datetime.now()
        if bDraw==None:
            bDraw = self.cbOptions1.lines[1][0].get_visible() or self.cbOptions1.lines[2][0].get_visible()

        if bDraw and self.Gate is not None:
            I = self.Mol.LocalTransmission(self.Gate+self.Mol.Efermi) if self.cbOptions1.lines[1][0].get_visible() else self.Mol.LocalCurrent(self.Gate+self.Mol.Efermi)
            if not self.bSplitLeads:
                I[:,1] = I[:,0]
                
            maxCurrent = abs(I).max()
            print ("Maximum current:", maxCurrent)
            
            for spin in range(2):
                for lead in range(1+self.bSplitLeads):
                    for i in range(self.Mol.N):
                        for j in range(i+1,self.Mol.N):
                            Inet = (I[spin, lead, j,i]-I[spin, lead, i,j])/2
                            color = self.cmapTrans(abs(Inet)) if self.cbOptions1.lines[1][0].get_visible() else self.cmapCurrent(abs(Inet))

                            if Inet>0:
                                self.LCPlot[spin, lead, i,j].set_linewidth(Inet/maxCurrent*10)
                                self.LCPlot[spin, lead, i,j].set_color(color)
                                self.LCPlot[spin, lead, i,j].set_visible(True)
                                self.LCPlot[spin, lead, j,i].set_visible(False)
                            elif Inet<0:
                                self.LCPlot[spin, lead, j,i].set_linewidth(-Inet/maxCurrent*10)
                                self.LCPlot[spin, lead, j,i].set_color(color)
                                self.LCPlot[spin, lead, j,i].set_visible(True)
                                self.LCPlot[spin, lead, i,j].set_visible(False)
                            elif i!=j and self.Mol.Ham[i,j]!=0:
                                self.LCPlot[spin, lead, i,j].set_visible(False)
                                self.LCPlot[spin, lead, j,i].set_visible(False)
        else:
            for spin in range(2):
                for lead in range(1+self.bSplitLeads):
                    for i in range(self.Mol.N):
                        for j in range(self.Mol.N):
                            if i!=j and self.Mol.Ham[i,j]!=0:
                                self.LCPlot[spin, lead, i,j].set_visible(False)
            
        print('   $GraphMolecule - DrawLocalCurrents:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def OnClickMolecule (self, event):
        if event.dblclick:
            self.UpdateAtomSel(None)

    def OnPickMolecule (self, event):
        self.UpdateAtomSel(event.ind[0], bPicked=True)

    def OnPress (self, event):
        if (self.AtomSel is None) or isinstance(self.AtomSel, np.ndarray): return
        if event.key in ('right', 'left'): 
            if event.key=='right': inc = 1
            else:  inc = -1
            self.UpdateAtomSel((self.AtomSel + inc)%len(self.Mol.Atom), True)
        

class GraphBias(object):
    """ Manages the displaying and control of the bias
        Attributes
        axBias                  Handle to the bias window

        Methods
        InitBias()
        DrawBias()              Draws the bias
        OnClickBias(event)      Changes the bias based on a click of the right handed side of the mouse
        """
    
    def InitBias (self):
        return
    
    def DrawBias (self):
        return #temporary
        start = datetime.datetime.now()
        self.axBias.cla()
        self.axBias.axhspan(-self.Mol.Bias/2, self.Mol.Bias, 0.00, 0.05, facecolor='y', alpha=0.5, zorder=3)
        self.axBias.axhspan(-self.Mol.Bias/2,         0, 0.95, 1.00, facecolor='y', alpha=0.5, zorder=3)
        self.axBias.set_ylim([-self.Mol.Bias/2 ,float(3)/2*self.Mol.Bias])
        self.axBias.set_ylabel('$Bias [V]$', color='y')
        for tl in self.axBias.get_yticklabels():
            tl.set_color('y')
        print('   $GraphBias - DrawBias:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def OnClickBias (self, event):
        if event.dblclick:
            self.UpdateBias(self.Bias*10)
        else:
            [ymin, ymax] = self.axMol.get_ylim()
            self.UpdateBias(-self.Bias/2+2*self.Bias*((event.ydata-ymin)/(ymax-ymin)))
        

class GraphTrans(object):
    """ Manages the displaying of the transmission and current and the control on the gate
        Attributes
        axTrans                 Handle to the transmission window
        GatePlot                Handle to the plot of the gate
        CurrentPlot             Handle to the plot of the current
        TransPlot               Handle to the plot of the transmisssion
        PhasePlot               Handle to the plot of the phase
        DOSPlot                 Handle tot the plot of the density of states

        Methods
        InitTrans()
        DrawTransmission()      Draws the transmission and its phase
        DrawCurrent()           Draws the current
        DrawDOS()               Draws the density of states
        DrawGate()              Draws the selected gate voltage
        OnClickTrans(event)     Changes the gate based on a click with the mouse
        """
    
    def InitTrans (self):
        start = datetime.datetime.now()
        (self.axTrans, self.axCurrent, self.axDOS, self.axOrb)  = [self.axTrans, self.axTrans.twinx(), self.axTrans.twinx(), self.axTrans.twinx()]        
        self.axCurrent.set_position (self.axTrans.get_position())   
        self.axDOS.set_position (self.axTrans.get_position())
        
        self.axCurrent.spines['right'].set_position(('axes', -0.07-1))
                        
        self.axDOS.spines['right'].set_position(('axes', -0.97))    #set to -0.07 to set on proper position
        self.axDOS.tick_params(axis='y', colors='darkorange')
        self.axDOS.set_ylabel("$DOS$", color='darkorange')
        
        self.GatePlot,   = self.axDOS.plot([1, 1],[1, 1], 'r', alpha=0.5, linewidth = 1, visible=False)
        self.CurrentPlot,= self.axCurrent.semilogy(1,1, 'b', visible=True)
        self.TransPlot,  = self.axTrans.semilogy(1,1, 'g', visible=True)
        
        self.DOSPlot = np.array([None, None, None])
        self.DOSPlot[0],    = self.axDOS.plot(1,1, 'darkorange', visible=True)
        self.DOSPlot[1],    = self.axDOS.plot(1,1, 'darkorange', ls=':', alpha=1.0, visible=True)
        self.DOSPlot[2],    = self.axDOS.plot(1,1, 'darkorange', ls=':', alpha=1.0, visible=True)
        
        self.axTrans.set_xlabel('$\epsilon-\epsilon_F (eV)$')
        
        print('   $GraphTrans - Init:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def DrawDOS (self, bDraw=None):
        start = datetime.datetime.now()
        if bDraw==None:
            bDraw = self.cbOptions2.lines[2][0].get_visible()
        self.DOSPlot[0].set_visible(bDraw)
        self.DOSPlot[1].set_visible(bDraw)
        self.DOSPlot[2].set_visible(bDraw)
        
        if bDraw:
            self.DOSPlot[0].set_data(self.Mol.Gates-self.Mol.Efermi, np.real(self.Mol.DOS.sum(axis=0)))
            self.DOSPlot[1].set_data(self.Mol.Gates-self.Mol.Efermi, np.real(self.Mol.DOS[0]))
            self.DOSPlot[2].set_data(self.Mol.Gates-self.Mol.Efermi, np.real(self.Mol.DOS[1]))
            
        #Plot tweaking
        self.axDOS.relim()
        self.axDOS.autoscale_view()
        self.axDOS.set_xlim([min(self.Mol.Gates-self.Mol.Efermi), max(self.Mol.Gates-self.Mol.Efermi)])
        
        #Set colorbar, also used as a reference for the LDOS
        norm = mpl.colors.Normalize(0, self.axDOS.get_ylim()[1])
        self.cbLDOS = mpl.colorbar.ColorbarBase(self.axLDOScb, cmap=self.cmapLDOS, norm=norm, orientation='vertical')
        self.cbLDOS.set_label('$DOS$', labelpad=-30)      
        
        print('   $GraphTrans - DrawDOS:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawTransmission (self, bDraw=None, bDrawPhase=None):
        start = datetime.datetime.now()
        if bDraw==None:
            bDraw = self.cbOptions2.lines[0][0].get_visible()
        self.TransPlot.set_visible(bDraw)

        if bDraw:
            self.TransPlot.set_data(self.Mol.Gates-self.Mol.Efermi, np.sum(self.Mol.T, axis=0))
            
        try:
            self.axTrans.relim()
            self.axTrans.autoscale_view()
            self.axTrans.set_xlim([min(self.Mol.Gates-self.Mol.Efermi), max(self.Mol.Gates-self.Mol.Efermi)])
        except:
            pass
        
        #Set colorbar, also used as a reference for the local transmission
        norm = mpl.colors.LogNorm(self.axTrans.get_ylim()[0], self.axTrans.get_ylim()[1])
        self.cbTrans = mpl.colorbar.ColorbarBase(self.axTranscb, cmap=self.cmapTrans, norm=norm, orientation='vertical')
        self.cbTrans.set_ticks([-0.01])
        self.cbTrans.set_label('$Transmission$', labelpad=-10)  

        print('   $GraphTrans - DrawTransmission:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def DrawCurrent (self, bDraw=None):
        start = datetime.datetime.now()
        if bDraw==None:
            bDraw = self.cbOptions2.lines[1][0].get_visible()
        self.CurrentPlot.set_visible(bDraw)
        
        if bDraw:            
            self.CurrentPlot.set_data(self.Mol.Gates-self.Mol.Efermi, abs(np.sum(self.Mol.I, axis=0)))
        
        self.axCurrent.relim()
        self.axCurrent.autoscale_view()
        self.axCurrent.set_xlim([min(self.Mol.Gates-self.Mol.Efermi), max(self.Mol.Gates-self.Mol.Efermi)])
        
        #Set colorbar, also used as a reference for the local transmission
        norm = mpl.colors.LogNorm(self.axCurrent.get_ylim()[0], self.axCurrent.get_ylim()[1])
        self.cbTrans = mpl.colorbar.ColorbarBase(self.axCurrentcb, cmap=self.cmapCurrent, norm=norm, orientation='vertical')
        self.cbTrans.set_ticks([-0.01])
        self.cbTrans.set_label('$Current \ [nA]$', labelpad=-15) 
      
        print('   $GraphTrans - DrawCurrent:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
                
    def DrawGate (self):
        if self.Gate == None or isinstance(self.OrbitalSel, int):
            self.GatePlot.set_visible(False)
        else:
            self.GatePlot.set_visible(True)
            self.GatePlot.set_data([self.Gate, self.Gate+self.Mol.Bias], [self.axDOS.get_ylim()])
        
    def OnClickTrans (self, event):
        if event.dblclick:
            self.UpdateGate(None)
        else:
            self.UpdateGate(event.xdata)
            
        
class GraphOrbitals (object):
    """ Manages the displaying of the eigenstates(orbitals?) of the molecule at the eigenvalues/energies
        Attributes
        axPhase                 Handle to the phase window
        axOrb                   Handle to the orbitals window
        plOrb                   Handle to the orbital plots
        rcOrb                   Handle to the rectangles of the orbital plots
        plSelOrb                Handle to the selected orbital plots
        rcSelOrb                handle to the rectangles of the selected orbital plots
        SelPhasePlot            Handle to the phase plot of the selected orbital
        SelTransPlot            Handle to the transmission plot of the selected orbital

        Methods
        InitTrans()
        ChangeMoleculeOrbitals  Creates a new number of handles for the orbitals and selected orbitals when the molecule changes
        DrawOrbitals()          Draws the eigenstates/orbitals at the appropriate eigenvalues/energies
        OnPickOrbital(event)    Changes the selected eigenstate/orbital
        DrawSelOrbitals()       Draws the indicators for the selected orbitals or series of orbitals
        DrawSelTransmission()   Draws the transmission and its phase from the selected orbital
        """

    def InitOrbitals (self):
        start = datetime.datetime.now()
        self.axOrb.set_position (self.axTrans.get_position())
        
        self.axOrb.spines['right'].set_position(('axes', 2.07)) 

        self.plOrb = [[],[]]
        self.rcOrb = [[],[]]
        self.plSelOrb = [[],[]]
        self.rcSelOrb = [[],[]]
        
        print('   $GraphOrbitals - Init:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def ChangeMoleculeOrbitals(self):
        start = datetime.datetime.now()
        for spin in range(2):
            for i in range(len(self.plOrb[spin])):
                self.rcOrb[spin][i].remove()
                self.rcSelOrb[spin][i].remove()
            
        del self.plOrb, self.rcOrb, self.plSelOrb, self.rcSelOrb
        self.plOrb = [[],[]]
        self.rcOrb = [[],[]]
        self.plSelOrb = [[],[]]
        self.rcSelOrb = [[],[]]
        
        for spin in range(2):
            for i in range(self.Mol.N):
                self.plOrb[spin].append(self.axOrb.bar(0, 0, picker=0.0, color='lightsteelblue', width=0.1, linewidth=1, alpha=0.4))
                if spin:    self.plOrb[spin][i][0].set_hatch('.')
                for rect in self.plOrb[spin][i]:  self.rcOrb[spin].append(rect)
                self.plSelOrb[spin].append(self.axOrb.bar(0, 0, color='r', width=0.1, linewidth=0, alpha=0.4))
                for rect in self.plSelOrb[spin][i]:   self.rcSelOrb[spin].append(rect)
                    
        print('   $GraphOrbitals - ChangeMolecule:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
    def DrawOrbitals (self, bDraw=None):
        start = datetime.datetime.now()
        ymax = 0
        if bDraw==None:
            bDraw = self.cbOptions2.lines[3][0].get_visible()
            
        if bDraw:
            width = self.rcOrb[0][1].get_width()
            for spin in range(1+self.Mol.bSpin):
                for i in range(self.Mol.N):
                    if i==0 or abs(self.Mol.e_arr[spin][i-1]-self.Mol.e_arr[spin][i])>width/2:
                        y = spin

                    self.rcOrb[spin][i].set_xy((self.Mol.e_arr[spin][i]-self.Mol.Efermi-width/2, y))
                    self.rcOrb[spin][i].set_height(1)
                    self.rcSelOrb[spin][i].set_xy((self.Mol.e_arr[spin][i]-self.Mol.Efermi-width/2, y))
                    self.rcSelOrb[spin][i].set_height(0)
                    y+=1+self.Mol.bSpin
                    ymax = max(ymax, y)
            if not self.Mol.bSpin:
                for i in range(self.Mol.N):
                    self.rcOrb[1][i].set_height(0)
                    self.rcOrb[1][i].set_xy((0,0))                    
        else:
            for spin in range(2):
                for i in range(self.Mol.N):
                    self.rcOrb[spin][i].set_height(0)
                    self.rcOrb[spin][i].set_xy((0,0))
            self.OrbitalSel = None

        self.axOrb.set_ylim([0, ymax*1.1])
        print('   $GraphOrbitals - DrawOrbitals:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def OnPickOrbital (self, event):
        print('Trying to pick')
        for spin in range(1+self.Mol.bSpin):
            for i, rect in enumerate(self.rcOrb[spin]):
                if rect==event.artist:
                    print('Found', spin, i)
                    return self.UpdateOrbitalSel(spin, i)

    def DrawSelOrbitals (self, bDraw=None):
        start = datetime.datetime.now()
        if bDraw==None:
            bDraw = self.cbOptions2.lines[3][0].get_visible()

        if np.shape(self.OrbitalSel)==(2,self.Mol.N) and bDraw:
            for spin in range(2):
                for i, height in enumerate(self.OrbitalSel[spin]):
                    self.rcSelOrb[spin][i].set_height(np.sqrt(abs(height)))
        else:
            for spin in range(2):
                for i in range(self.Mol.N):
                    self.rcSelOrb[spin][i].set_height(0)
            if self.OrbitalSel!=None and bDraw:
                self.rcSelOrb[self.OrbitalSel[0]][self.OrbitalSel[1]].set_height(1)

        print('   $GraphOrbitals - DrawSelOrbitals:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')



class GraphIVCurve (object):
    """ Manages the displaying of the current versus bias curve at a certain energy
        Attributes
        axIVCurve               Handle to the IV Curve window
        Biases                  Array of biases to be calculated
        IVPlot                  Handle to the plot of the IVCurve

        Methods
        InitIVCurve()
        DrawIVCurve()           Draws the current against the applied voltage
        """
    def InitIVCurve (self):
        self.Biases = np.linspace(-2, 2, 100)
        self.IVPlot,  = self.axIVCurve.plot(1,1, 'b', visible=True)
        self.axIVCurve.set_ylabel('$Current \ [nA]$')
        self.axIVCurve.set_xlabel('$Bias \ [V]$')

    def DrawIVCurve (self):
        start = datetime.datetime.now()
        if self.Gate==None:
            self.IVPlot.set_visible(False)
            print('   $GraphiIVCurve - DrawIVCurve (False):', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
            return
        
        self.IVPlot.set_visible(True)
        self.IVPlot.set_data(self.Biases, np.sum(np.real(self.Mol.CurrentAnalytical(self.Gate*np.ones(len(self.Biases)), self.Gate+self.Biases)), axis=0))
        self.axIVCurve.relim()
        self.axIVCurve.autoscale_view()
        
        print('   $GraphiIVCurve - DrawIVCurve (True):', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

        
class Graphics (GraphMolecule, GraphBias, GraphTrans, GraphOrbitals, GraphIVCurve):
    """ Manages the graphical representation of the project
        Attributes
        fig                     Handle to the entire figure
        Bias                    The selected bias
        Gate                    The selected gate voltage
        OrbitalSel              The selected orbital or series of orbitals

        axSC                    Handle to the radio buttons window for the self consistency
        cbSC                    Handle to the radio buttons for the self consistency

        axOptions1              Handle to the checkbox window with options 1
        cbOptions1              Handle to the checkboxes with options 1
        
        axOptions2              Handle to the checkbox window with options 2
        cbOptions2              Handle to the checkboxes with options 2

        axGleft, axGright       Handle to the slider windows for selecting the lead interaction strength
        sGleft,  sGright        Handle to the sliders for selecting the lead interaction strength

        axSave                  Handle to the save button window
        bSave                   Handle to the save button

        Methods
        init()
        OnPick(event)           Manages the pick events
        OnClick(event)          Manages the on click events
        Save()                  Manages the input from the save button
        Options1(label)         Manages the input from the options 1 window
        Options2(label)         Manages the input from the options 2 window

        SetMolecule(Mol)        Sets the molecule class from which the graphics gets its information
        UpdateMolecule()        Updates everything that changes when the molecule has changed

        UpdateConsistency(label)Updates the selected consistency method
        UpdateG(val)            Updates the interaction strength with the leads
        UpdateBias(bias)        Updates the selected bias
        UpdateHamExt()          Updates everything that changes when the extended hamiltonian is changed
        
        UpdateGate(gate)        Updates the selected gate voltage
        UpdateAtomSel(iAtom)    Updates the selected atom
        UpdateOrbitalSel(iOrb)  Updates the selected orbital
        UpdateSelection()       Updates everything that changes after one of the selections has changed

    """

    def __init__(self):
        start = datetime.datetime.now()
        #General parameters
        self.bGraph = True
        self.bSplitLeads = False
        
        #Set up initial parameters
        self.Gate = None
        self.OrbitalSel = None
        
        #Set up the plots and their positions
        self.fig, (self.axBias, self.axTrans, self.axIVCurve) = plt.subplots(3,1)
        self.fig.patch.set_facecolor('ghostwhite')
        plt.subplots_adjust(left=0.3)
        pos = self.axBias.get_position()
        self.axBias.set_position ([pos.x0, 0.55, pos.width, 0.40])
        self.axTrans.set_position([pos.x0, 0.05, pos.width, 0.40])
        self.axIVCurve.set_position([0.05, 0.05, 0.15, 0.3])
        
        #Set up the colorbars
        c = mcolors.ColorConverter().to_rgb
        self.axLDOScb    = self.fig.add_axes([pos.x0+pos.width-0.01-1, 0.05, 0.01, 0.4])
        self.cmapLDOS    = make_colormap([c('darkgoldenrod'), c('darkorange'), 0.50, c('darkorange'), c('palegoldenrod')])
        self.axCurrentcb = self.fig.add_axes([pos.x0-0.051-1, 0.05, 0.01, 0.4])
        self.cmapCurrent = make_colormap([c('navy'),          c('dodgerblue'), 0.50, c('dodgerblue'), c('skyblue')])
        self.axTranscb   = self.fig.add_axes([pos.x0, 0.05, 0.01, 0.4])
        self.cmapTrans   = make_colormap([c('g'), c('limegreen')])
        self.axLDcb      = self.fig.add_axes([pos.x0-0.05, 0.55, 0.01, 0.4])
        self.cmapLD      = make_colormap([c('thistle'), c('mediumvioletred')])
#        self.axPotcb     = self.fig.add_axes([pos.x0, 0.55, pos.width, 0.01], zorder=5)
#        self.cmapPot     = make_colormap([c('black'), c('azure')])
        
        #Connect clicking and keyboard events
        self.fig.canvas.mpl_connect('button_press_event', self.OnClick)
        self.fig.canvas.mpl_connect('pick_event', self.OnPick)
        self.fig.canvas.mpl_connect('key_press_event', self.OnPress)

        #Initialize the individual plots
        self.InitMolecule()
        self.InitBias()
        self.InitTrans()
        self.InitOrbitals()
        self.InitIVCurve()
        
        #Set up the option boxes
        self.axSC = plt.axes([0.05, 0.85, 0.15, 0.10], axisbg='white')
        self.cbSC = RadioButtons(self.axSC, ('Not self consistent', 'Hubbard', 'PPP'))
        self.cbSC.on_clicked(self.UpdateConsistency)

        self.axOptions0 = plt.axes([0.05, 0.76, 0.15, 0.07], axisbg='white')
        self.cbOptions0 = CheckButtons(self.axOptions0, ('Overlap', 'Spin'), (False, False))
        self.cbOptions0.on_clicked(self.Options0)
        
        self.axOptions1 = plt.axes([0.05, 0.64, 0.15, 0.10], axisbg='white')
        self.cbOptions1 = CheckButtons(self.axOptions1, ('Show Local Density','Show Local Transmission', 'Show Local Currents'), (False, False, True))
        self.cbOptions1.on_clicked(self.Options1)
        
        self.axOptions2 = plt.axes([0.05, 0.47, 0.15, 0.15], axisbg='white')
        self.cbOptions2 = CheckButtons(self.axOptions2, ('Show Transmission', 'Show Current', 'Show DOS', 'Show Orbitals'), (False, False, True, False))
        self.cbOptions2.on_clicked(self.Options2)
        c = ['seagreen', 'b', 'darkorange', 'lightsteelblue']    
        [rec.set_facecolor(c[i]) for i, rec in enumerate(self.cbOptions2.rectangles)]
        
        self.axGam_L  = plt.axes([0.05, 0.43, 0.15, 0.02], axisbg='white')
        self.axGam_R  = plt.axes([0.05, 0.40, 0.15, 0.02], axisbg='white')
        self.axGam_b  = plt.axes([0.05, 0.37, 0.15, 0.02], axisbg='white')
        self.sGam_L   = Slider(self.axGam_L, '$\Gamma^S [eV]$', 0.0, 1.0, valinit = Gam_L, color = 'y')
        self.sGam_R   = Slider(self.axGam_R, '$\Gamma^D [eV]$', 0.0, 1.0, valinit = Gam_R, color = 'y')
        self.sGam_b   = Slider(self.axGam_b, '$\Gamma^M [eV]$', 0.0, 0.2, valinit = Gam_b, color = 'k')
        self.sGam_L.on_changed(self.UpdateGam_LR)
        self.sGam_R.on_changed(self.UpdateGam_LR)
        self.sGam_b.on_changed(self.UpdateGam_b)
               
        print('   $Graph - Init:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def OnPick(self, event):
        if isinstance(event.artist, Rectangle):
            self.OnPickOrbital(event)
        else:
            self.OnPickMolecule(event)
                   
    def OnClick (self, event):
        if event.inaxes==self.axMol:
            if event.button==1:
                self.OnClickMolecule (event)
            elif event.button==3:
                self.OnClickBias (event)
        elif event.inaxes==self.axOrb:
            if event.button==1:
                return
            if event.button==3:
                self.OnClickTrans (event)
        return
        
    def Options0(self, label):
        if label == 'Overlap':
            self.Mol.SetOverlap(self.cbOptions0.lines[0][0].get_visible())
            self.UpdateMolecule()
        elif label == 'Spin':
            self.Mol.SetSpin(self.cbOptions0.lines[1][0].get_visible())
            self.UpdateSystem()
        return
        
    def Options1(self, label):
        if label == 'Show Local Density':
            self.DrawLocalDensity(self.cbOptions1.lines[0][0].get_visible())
        elif label == 'Show Local Transmission':
            self.cbOptions1.lines[2][0].set_visible(False)
            self.DrawLocalCurrents(self.cbOptions1.lines[1][0].get_visible())
        elif label == 'Show Local Currents':
            self.cbOptions1.lines[1][0].set_visible(False)
            self.DrawLocalCurrents(self.cbOptions1.lines[2][0].get_visible())
        self.fig.canvas.draw()
        return

    def Options2(self, label):
        if label == 'Show Transmission':
            self.DrawTransmission()
        elif label == 'Show Current':
            self.DrawCurrent()
        elif label == 'Show DOS':
            self.DrawDOS()
        elif label == 'Show Orbitals':
            self.DrawOrbitals()
            self.DrawSelOrbitals()
            
        self.fig.canvas.draw()
        return

     
    def SetMolecule(self, Molecule):
        self.Mol = Molecule
        self.cbOptions0.lines[0][0].set_visible(self.Mol.bOverlap)
        self.cbOptions0.lines[1][0].set_visible(self.Mol.bSpin)
        self.UpdateMolecule(bSet=True)
        
    def UpdateMolecule (self, bSet=False):
        if not bSet:
            self.Mol.UpdateMolecule()
        
        if self.bGraph:
            self.ChangeMolecule ()
            self.ChangeMoleculeOrbitals ()
            self.DrawMolecule ()
        return self.UpdateSystem ()
        

    def UpdateConsistency(self, label):
        self.Mol.SetConsistency(label)
        return self.UpdateSystem ()     
       
    def UpdateGam_LR (self, val):
        self.Mol.SetGam_LR (self.sGam_L.val, self.sGam_R.val)
        return self.UpdateSystem ()
        
    def UpdateGam_b(self, val):
        self.Mol.SetGam_b(self.sGam_b.val)
        return self.UpdateSystem()
    
    def UpdateBias(self, bias):
        self.Mol.UpdateBias(bias)
        return self.UpdateSystem ()

    def UpdateSystem (self):        
        if self.bGraph:
            self.DrawLocalDensity ()
#        self.DrawPotential()
            self.DrawBias ()
            self.DrawTransmission ()
            self.DrawCurrent ()
            self.DrawDOS ()
            self.DrawOrbitals ()
        return self.UpdateSelection()

    def UpdateGate (self, gate):
        self.Gate = gate
        print ("Gates voltage set to: ", self.Gate, "eV")
        self.OrbitalSel = None
        self.AtomSel    = self.Mol.LDOS(gate+self.Mol.Efermi if gate!=None else gate)
        return self.UpdateSelection()

    def UpdateAtomSel(self, iAtom, bPicked=False):
        self.AtomSel = iAtom
        if self.AtomSel == None:
            print ("No atom selected")
            self.axMol.set_title('No atom selected')
            self.OrbitalSel = None
        elif bPicked:
            print ("Selected atom", self.Mol.Atom[self.AtomSel][0], "at", self.AtomSel, " Local density = ", self.Mol.LD[0][self.AtomSel])
            self.axMol.set_title('Selected atom: %c at %d. Density = %f'%(AtomNr[self.Mol.Atom[self.AtomSel][0]],self.AtomSel, self.Mol.LD[0][self.AtomSel]))
            self.OrbitalSel = np.real(self.Mol.eigvec[:,self.AtomSel, :]*np.conjugate(self.Mol.eigvec[:,self.AtomSel, :]))
            self.Gate = None
        return self.UpdateSelection()
        
    def UpdateOrbitalSel(self, spin, iOrbital):
        self.OrbitalSel = [spin, iOrbital]
        
        if self.OrbitalSel!=None:
            print ("Orbital set to:", self.OrbitalSel, "with energy", self.Mol.e_arr[spin][iOrbital]-self.Mol.Efermi, "eV")
            sel = np.real(self.Mol.eigvec[spin][:,iOrbital]*np.conjugate(self.Mol.eigvec[spin][:,iOrbital]))
            self.AtomSel    = np.array([np.zeros(self.Mol.N), sel]) if spin else np.array([sel, np.zeros(self.Mol.N)])
            self.Gate       = self.Mol.e_arr[spin][iOrbital]
        return self.UpdateSelection()

    def UpdateSelection (self):
        if self.bGraph:
            self.DrawLocalCurrents ()
            self.DrawGate()
            self.DrawIVCurve()
            self.DrawLDOS()
            self.DrawSelOrbitals()
        
            self.fig.canvas.draw()
        return True
        
        
    def Save(self, szFileName, part=None):
        self.fig.set_size_inches(10.5, 6.5)
        if part==None:
            self.fig.savefig(szFileName,  dpi=self.fig.dpi)
            
        if part=='Molecule':
            self.fig.set_size_inches(15.5, 30.5)
            self.axMol.axis('equal')
            self.axMol.relim()
            self.axMol.autoscale_view()
            extent = self.axBias.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(szFileName,  bbox_inches=extent.expanded(0.95, 0.95))           
        return
        
        
                   
            

if __name__ == '__main__':
    from Molecule import *
    from AtomParam import cAtom
    

    Mol = AGNR(5, 101)
    Mol.SetGam_b(0.1)
    Mol.SetLeads()
    Mol.UpdateMolecule()
    Mol.UpdateGates(-3.5+Mol.Efermi, 3.5+Mol.Efermi)
    
    Graph = Graphics()
    Graph.SetMolecule(Mol)
    
    
    plt.show()

