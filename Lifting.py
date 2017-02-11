# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:34:26 2016

@author: timge
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

from AtomParam import *
from ImportExport import *
from ShowMD import *

l = 1.42

class MD_Lifting(object):
    def __init__(self, Mol):
        self.Mol = Mol
        
        self.Ebinding = -0.040 #eV per atom
        self.Ebending =  3.9 #eV per atom per length*
        self.Force    = 0.1 #eV/Angstrom
        
    def SetEbinding(self, Ebinding):
        self.Ebinding = Ebinding
        
    def SetEbending(self, Ebending):
        self.Ebending = Ebending
        
    def SetForce(self, Force):
        self.Force = Force
        
    def GetConfiguration(self, L, dz, R):
        N = int(L/1.42)*10
        r = np.zeros((3, N))
        r[0] = np.linspace(0, L, N)

        if dz>L:
            return r
    
        if R<dz and dz-R+np.pi/2*R<L:
#            print('Situation 1')
            for i in range(N):
                if r[0, i] < dz-R:
                    r[2, i]=dz-r[0, i]
                elif r[0, i] < dz-R+np.pi/2*R :
                    theta = (r[0,i]-(dz-R))/(R*np.pi/2)*np.pi/2
                    r[1, i]= R - R*np.cos(theta)
                    r[2, i]= R - R*np.sin(theta)
                else:
                    r[1, i] = r[0, i]-(dz-R)-np.pi/2*R+R
                
            BendingLimits = [dz-R, dz-R+np.pi/2*R]
                
        elif dz<=R and np.arccos(1-dz/R)*R<L:
#            print('Situation 2')
            theta = np.arccos(1-dz/R)
            X0 = np.sin(theta)*R
            for i in range(N):
                if r[0,i] < theta/(2*np.pi)*2*np.pi*R:
                    phi = theta-r[0,i]/(2*np.pi*R)*(2*np.pi)
                    r[1, i] = X0-np.sin(phi)*R
                    r[2, i] = R -np.cos(phi)*R
                else:
                    r[1, i] = X0 + r[0,i] - theta/(2*np.pi)*2*np.pi*R
            BendingLimits=[0, theta/(2*np.pi)*2*np.pi*R]
        
        else: # False:# L/R<np.pi/2:
#            print('Situation 3')
            theta = L/R
            alpha = np.linspace(0, np.pi/2, 100) 
            
            if np.min(np.sin(alpha+theta)-np.sin(alpha)) <= dz/R <= np.max(np.sin(alpha+theta)-np.sin(alpha)):
                alpha = alpha[np.argmin(abs(np.sin(alpha+theta)-np.sin(alpha)-dz/R))]
                X0 = R-R*np.cos(alpha)
                Y0 = R*np.sin(alpha+theta)        
            
                for i in range(N):
                    phi = alpha + r[0,i]/R
                    r[1, i]= R-R*np.cos(phi)-X0
                    r[2, i]= Y0-R*np.sin(phi)
                BendingLimits = [0, L]
            else:
#                print('Situation 4')
                phi = np.linspace(0, np.pi/2, 100)
                phi = phi[np.argmin(abs(L-phi*R-dz+np.sin(phi)*R))]
                h = dz-np.sin(phi)*R
                for i in range(N):
                    if r[0,i] < h:
                        r[2, i] = dz-r[0,i]
                    else:
                        theta = (r[0, i]-h)/R
                        r[1, i] = R-np.cos(theta)*R
                        r[2, i] = dz-h-np.sin(theta)*R
                BendingLimits=[h, L]                    
                
        return r, BendingLimits
        
    def GetConfigurationOptimum(self, dz, ax=None, bExport=False, xEnd=None, xFCC=None, export=[]):
        start = datetime.datetime.now()
        N = 100
        
        x = self.Mol.Atom[:,1]
        y = self.Mol.Atom[:,2]
        z = self.Mol.Atom[:,3]
        
        L = np.max(x)-np.min(x)
        
        def GetBindingEnergy(r):        
            xx, x0 = np.meshgrid(x, r[0])
            X = r[1][np.argmin(abs(xx-x0), axis=0)]
            Z = r[2][np.argmin(abs(xx-x0), axis=0)]
        
            if ax!=None:
#                ax[0].plot(X, Z, 'ok')
                ax[0].plot(r[1], r[2], 'k')
            
            return np.sum(np.exp(-Z/1.5))*self.Ebinding
        
        def GetBendingEnergy(r, BendingLimits, R):            
            N = np.sum(-1/(np.exp((x-BendingLimits[0])*4)+1) + 1/(np.exp((x-BendingLimits[1])*4)+1)   )
            return N*self.Ebending/((R)**2)/2        
            
        def GetFrictionEnergy(r, xEnd):
            if xEnd==None:
                return 0
            return self.Force*abs(r[1,-1]-xEnd)
                                
        R = np.linspace((L/20)**(1/10), (L*10)**(1/10), N)**10
        
        Ebind = np.zeros(len(R))
        Ebend = np.zeros(len(R))
        Efric = np.zeros(len(R))
        
        for i, r in enumerate(R):            
            r, BendingLimits = self.GetConfiguration(L, dz, r)
            Ebind[i] = GetBindingEnergy(r)
            Ebend[i] = GetBendingEnergy(r, BendingLimits, R[i])
            Efric[i] = GetFrictionEnergy(r, xEnd)
            
            if 'MD' in export:
                xx, x0 = np.meshgrid(x, r[0])    
                X = r[1][np.argmin(abs(xx-x0), axis=0)]
                Y = y
                Z = r[2][np.argmin(abs(xx-x0), axis=0)] 
                ShowConfiguration(np.array([self.Mol.Atom[:,0], X, Y, Z]), STM=1, szFileNameSave=self.Mol.szOutputFolder + self.Mol.Name + ' - dz=' + str(dz) + ' r=' + str(R[i]) + '.png')
                


        ind = np.argmin(Ebind+Ebend+Efric)
                
        if ax!=None:
            ax[1].semilogx(R, Ebind, 'g')
            ax[1].semilogx(R, Ebend, 'b')
            ax[1].semilogx(R, Efric, 'r')
            ax[1].semilogx(R, Ebind+Ebend+Efric, 'k')
            ax[1].semilogx(R[ind], (Ebind+Ebend+Efric)[ind], 'r*', lw=0)
        
        r, BendingLimits = self.GetConfiguration(L, dz, R[ind]) 
        print('   $Search MD optimum for z=', int(10*dz)/10, ':', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        
        if 'Energy' in export:
            header = ['R', 'Bending', 'Binding', 'Friction']
            values = np.zeros((4, len(R)))
            values[0] = R
            values[1] = Ebend
            values[2] = Ebind
            values[3] = Efric
            ExportData(self.Mol.szOutputFolder + self.Mol.Name + ' - dz=' +  str(dz) + ' energies.txt', np.swapaxes(values, 0, 1), header)
        
        if bExport:  
            xx, x0 = np.meshgrid(x, r[0])    
            X = r[1][np.argmin(abs(xx-x0), axis=0)]
            Y = y
            Z = r[2][np.argmin(abs(xx-x0), axis=0)]    
            
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(X, Y, Z, c='k')
            ax.set_xlim([0, 1.1*L])
            ax.set_ylim([0, 1.1*L])
            ax.set_zlim([0, 1.1*L])      
                        
        return r
            
    def PerformLiftingMD(self, DZ, bPlot = True, export=[]):
        fig = plt.figure()
        x = self.Mol.Atom[:,1]
        y = self.Mol.Atom[:,2]
        z = self.Mol.Atom[:,3]
        
        L = np.max(x)-np.min(x)
        
        DX = np.zeros(len(DZ))
        N  = np.zeros(len(DZ))
        
        if bPlot:
            axLift3D = fig.add_subplot(2, 3, 1, projection='3d')
            axLift2D = fig.add_subplot(2, 3, 2)
            axcbar = fig.add_axes([0.35, 0.535, 0.01, 0.37])
            norm = mpl.colors.Normalize(0, np.max(DZ))
            cmap = mpl.cm.get_cmap('viridis')
            cbar = mpl.colorbar.ColorbarBase(axcbar, cmap=cmap, norm=norm, orientation='vertical')
            cbar.set_label('$\Delta z$')
            self.cmap = cmap       
        
        xEnd = None
        
        for i, dz in enumerate(DZ):   
            if bPlot:
                ax1 = fig.add_subplot(4,len(DZ),i+1+2*len(DZ))
                ax2 = fig.add_subplot(4,len(DZ),i+1+3*len(DZ))  
                
            r = self.GetConfigurationOptimum(dz, ax=[ax1, ax2] if bPlot else None, xEnd = xEnd, export=['Energy'] if 'Energy' in export else [])
            xEnd = np.max(r[1])
            
            if 'xyz' in export:
                xx, x0 = np.meshgrid(x, r[0])   
                xyz = np.zeros((self.Mol.N,3))
                xyz[:,0] = r[1][np.argmin(abs(xx-x0), axis=0)]
                xyz[:,1] = self.Mol.Atom[:,2]
                xyz[:,2] = r[2][np.argmin(abs(xx-x0), axis=0)]
                ExportXYZ(self.Mol, i, xyz, dz)
                if 'MD' in export:
                    atom_xyz = np.zeros((4, self.Mol.N))
                    atom_xyz[0] = self.Mol.Atom[:,0]
                    atom_xyz[1:4] = xyz.T
                    ShowConfiguration(atom_xyz, STM=1, szFileNameSave=self.Mol.szOutputFolder + 'MD/' + self.Mol.Name + ' - dz=' + str(dz) + '.png')
                    
            
            if bPlot:
                ax1.plot(r[1], r[2], c=cmap(dz/np.max(DZ)), lw=2)
                ax1.set_xlim([-10, 1.1*L])
                ax1.set_ylim([0, 1.1*L])
            
                xx, x0 = np.meshgrid(x, r[0])    
                X = r[1][np.argmin(abs(xx-x0), axis=0)]
                Y = y
                Z = r[2][np.argmin(abs(xx-x0), axis=0)]    
                axLift2D.plot(r[1], r[2], c=cmap(dz/np.max(DZ)))
                axLift2D.scatter(X, Z, c=cmap(dz/np.max(DZ)), s=10, lw=0)
                axLift3D.scatter(X, Y, Z, c=cmap(dz/np.max(DZ)), lw=0, s=10, alpha=0.7)
            
                ax2.set_xlabel('$\Delta z = %0.1f$' % (dz))
            
                if i==0:
                    ax1.set_ylabel('Position')
                    ax2.set_ylabel('Energy [eV]')
                    ax2.legend(['Bending energy', 'Bonding energy', 'Friction', 'Total energy'])
                
                z =np.argwhere(r[2]>1.5)
                N[i] = np.sum(np.exp(-Z/1.5))
                DX[i] = 0 if len(z)==0 else r[0][np.max(z)]
                axLift2D.plot(r[1, 0 if len(z)==0 else np.max(z)], 0.1, '*', c=cmap(dz/np.max(DZ)), lw=0)
            
        if bPlot:
            axLift3D.set_xlim([0, 1.0*L])
            axLift3D.set_ylim([0, 1.0*L])
            axLift3D.set_zlim([0, 1.0*L])
            
            axDX = fig.add_subplot(2,3,3)
            axDX.plot(DZ, DX, c='k')
            axDX.set_xlim(0, np.max(dz))
            axDX.set_xlabel('$\Delta z$')
            axDX.set_ylim(0, L*1.1)
            axDX.set_ylabel('$\Delta x$')
            axDX.scatter(DZ, DX, c=cmap(DZ/np.max(DZ)), lw=0, s=50)
            axN = axDX.twinx()
            axN.scatter(DZ, N, c=cmap(DZ/np.max(DZ)), lw=0, s=50, marker='*')
            axN.set_ylabel('# connected atoms')
        
        if 'dx-n' in export:
            header = ['dz', 'dx', 'N']
            values = np.zeros((len(DZ), 3))
            values[:,0] = DZ
            values[:,1] = DX
            values[:,2] = N
            ExportData(self.Mol.szOutputFolder + '/' + self.Mol.Name + ' - dz-dz-N.txt', values, header)
        
    def PerformLifting(self, DZ, ax=None, export=[], axOrb=None):
        #Performs a lifing experiment where the lead lays on the left contact and is being pulled up by a right contact
        xEnd = np.max(self.Mol.Atom[:,1])
        
        I = np.zeros((len(DZ)))
        
        cmap = mpl.cm.get_cmap('viridis')
        
        for i, dz in enumerate(DZ):
            r = self.GetConfigurationOptimum(dz, xEnd=xEnd)
            xEnd = np.max(r[1])
            xx, x0 = np.meshgrid(self.Mol.Atom[:,1], r[0])
            self.Mol.SetSurface(r[2][np.argmin(abs(xx-x0), axis=0)])
            
            self.Mol.CreateHamExt()
            self.Mol.CalcGamma()
            self.Mol.Current()
            
            I[i] = -np.sum(self.Mol.CurrentAnalytical(np.array([self.Mol.Efermi+self.Mol.Bias]), np.array([self.Mol.Efermi])))
#            I[i] =  self.Mol.I[0][np.argmin(abs(self.Mol.Gates-self.Mol.Efermi))]             
            print ('Sequence at', i+1, '/', len(DZ), ',dz', dz)
            
            if 'xyz' in export:
                xyz = np.zeros((self.Mol.N,3))
                xyz[:,0] = r[1][np.argmin(abs(xx-x0), axis=0)]
                xyz[:,1] = self.Mol.Atom[:,2]
                xyz[:,2] = r[2][np.argmin(abs(xx-x0), axis=0)]
                ExportXYZ(self.Mol, i, xyz, dz)
            if 'DOS' in export:
                ExportDOS(self.Mol, self.Mol.szOutputFolder + self.Mol.Name + ' - DOS at %0.0f dz= %0.2f.txt' % (i, dz))
                
            if axOrb!=None:
#                axTrans.semilogy(self.Mol.Gates-self.Mol.Efermi, self.Mol.T, c=cmap(dz/np.max(DZ)))
                axOrb.plot(dz, self.Mol.e_arr[0][self.Mol.N/2-2]-self.Mol.Efermi, 'lightgreen', marker='.')
                axOrb.plot(dz, self.Mol.e_arr[0][self.Mol.N/2-1]-self.Mol.Efermi, 'g.')
                axOrb.plot(dz, self.Mol.e_arr[0][self.Mol.N/2-0]-self.Mol.Efermi, 'r.')
                axOrb.plot(dz, self.Mol.e_arr[0][self.Mol.N/2+1]-self.Mol.Efermi, 'fuchsia', marker='.')
#                axTrans.plot(dz, self.Mol.Efermi, 'k.')
        
        if ax!=None:
            ax.plot(DZ, np.log(I))
            
        if len(export)>0:
            if 'param' in export:
                ExportParameters(self.Mol)
            if 'dzI' in export:
                ExportData(self.Mol.szOutputFolder + self.Mol.Name + ' - Lifting vs Current.txt', np.swapaxes(np.concatenate((DZ, I)).reshape(2,len(DZ)),0,1), ['dz(Ang)', 'I(nA)'])
            
        return I
        
    def PerformLiftingSpectroscopy(self, DZ, Bias, ax=None, export=[]):
        #Performs a lifing experiment where the lead lays on the left contact and is being pulled up by a right contact and increases the bias on every height
        xEnd = np.max(self.Mol.Atom[:,1])
        
        I = np.zeros((len(DZ), len(Bias)))
        
        
        for i, dz in enumerate(DZ):
            r = self.GetConfigurationOptimum(dz, xEnd=xEnd)
            xEnd = np.max(r[1])
            xx, x0 = np.meshgrid(self.Mol.Atom[:,1], r[0])
            self.Mol.SetSurface(r[2][np.argmin(abs(xx-x0), axis=0)])
            
            self.Mol.CreateHamExt ()
            self.Mol.CalcGamma()
            
            I[i] = np.sum(self.Mol.CurrentAnalytical(self.Mol.Efermi*np.ones(len(Bias)), self.Mol.Efermi+Bias), axis=0)
            print ('Sequence at', i+1, '/', len(DZ), ',dz', dz)
            
            if 'xyz' in export:
                xyz = np.zeros((self.Mol.N,3))
                xyz[:,0] = r[1][np.argmin(abs(xx-x0), axis=0)]
                xyz[:,1] = self.Mol.Atom[:,2]
                xyz[:,2] = r[2][np.argmin(abs(xx-x0), axis=0)]
                ExportXYZ(self.Mol, i, xyz, dz)
          
            
        if len(export)>0:
            if 'param' in export:
                ExportParameters(self.Mol)
            if 'dz-V-I' in export:
                header = ['dz\Bias']
                for bias in Bias: header.append('%0.4f' % bias)
                data = np.zeros((len(DZ), len(Bias)+1))
                data[:,0] = DZ
                data[:,1:] = I
                ExportData(self.Mol.szOutputFolder + self.Mol.Name + ' - LiftingSpectroscopy.txt', data, header)
            
        return I


if __name__ == '__main__':
    from Molecule import *
    from AtomParam import cAtom
    
    Mol1 = Junction_5_7_AGNR([[5,5]])
    MD = MD_Lifting(Mol1)
    MD.PerformLiftingMD(np.linspace(0,40,41))
    
    plt.show()
