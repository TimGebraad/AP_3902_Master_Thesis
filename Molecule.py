import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
import pickle

##from tkinter import filedialog

from Green import *
from Graphics import *
from AtomParam import AtomNr, e0, tm, tm2, Overlap, Gam_S
#from Lifting import MD_Lifting
import math
import copy
import datetime

class Molecule (Green):
    """ Manages the description of the molecule
        Attributes
        Atom                    List with the atoms of the molecule
        lpL, lpR                List with the atoms that are coupled to the leads
        S                       Overlap matrix
        V                       Long range interaction matrix for PPP model
        
        Methods
        init
        Load(filename)          Loads a molecule from a file
        Save()                  Saves the current molecule to a file
        AddAtom(Atom, x, y, z)  Adds a certain atom at a specific position to the molecule
        UpdateAtom (iAtom, Atom)Changes the kind of atom at a specific position
        SetLead(iAtom, side)    Couples a specific atom to one of the leads
        SetLeads(lim)           Couples the atoms within a certain limit to the leads
        CreateHam()             Creates the tight binding Hamiltonian
        CreateOverlap()         Creates the overlap matrix
        CreateV()               Creates the long range interaction matrix for the PPP model
        GetCoord()              Returns vectors with the atom positions
        """
    def __init__ (self):
        self.N = 0
        self.Atom = np.zeros((0,4))
        self.Parts = []
        self.Name = ''
        self.szOutputFolder = ""
        super().__init__()
        
    def SetOutputFolder(self, szOutputFolder):
        self.szOutputFolder = szOutputFolder

    def AddAtom (self, Type, x, y, z):
        self.Atom = np.vstack((self.Atom, np.array([ {v: k for k, v in AtomNr.items()}[Type], x, y, z])))
        self.N = len(self.Atom)

    def UpdateAtom (self, iAtom, Atom):
        if Atom=='DEL':
            self.Atom = np.delete(self.Atom, iAtom, axis=0)
            self.N -=1
        else:
            self.Atom[iAtom,0] = {v: k for k, v in AtomNr.items()}[Atom]
                    
    def Translate(self, atoms, dr):
        self.Atom[atoms,1] +=dr[0]
        self.Atom[atoms,2] +=dr[1]
        self.Atom[atoms,3] +=dr[2]

    def Rotate (self, atoms, alpha):
        for i, (Atomi, xi, yi, zi) in enumerate(self.Atom[atoms]):
            new = (Atomi, math.cos(alpha)*xi-math.sin(alpha)*yi, math.sin(alpha)*xi+math.cos(alpha)*yi, zi)
            self.Atom[atoms[i]] = new

    def CreateHam (self, bOrder=1):
        start = datetime.datetime.now()
        self.N = len(self.Atom)
        
        """Create Tight Binding Hamiltonian"""
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        
        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        R2 = (x1-x2)**2+(y1-y2)**2  
  
        self.Ham = np.zeros((self.N, self.N), np.dtype('c16'))
        for i, (Typei, xi, yi, zi) in enumerate(self.Atom):
            Atomi = AtomNr[Typei]
            for j, (Typej, xj, yj, zj) in enumerate(self.Atom[0:i+1]):
                Atomj = AtomNr[Typej]
                if i==j:
                    self.Ham[i,j] = e0[Atomi]
                elif R2[i,j]<=(1.6*dL)**2:
                    if Atomi+Atomj in tm:
                        self.Ham[i,j] = tm[Atomi+Atomj]
                        self.Ham[j,i] = tm[Atomi+Atomj]
                elif R2[i,j]<=(1.8*dL)**2 and bOrder>1:
                    if Atomi+Atomj in tm2:
                        self.Ham[i,j] = tm2[Atomi+Atomj]
                        self.Ham[j,i] = tm2[Atomi+Atomj]

        """Introduce broadening"""
        self.Ham -= self.Gam_b*1j/2*np.identity(self.N)
                        
        print('   $Create Hamiltonian:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return self.Ham

    def CreateOverlap (self):
        start = datetime.datetime.now()
        if self.bOverlap:            
            x = self.Atom[:,1]
            y = self.Atom[:,2]
            
            x1, x2 = np.meshgrid(x, x)
            y1, y2 = np.meshgrid(y, y)
            R2 = (x1-x2)**2+(y1-y2)**2
            self.S = (R2 < (1.3*dL)**2)*Overlap
            np.fill_diagonal(self.S, 1)
        else:
            self.S = np.identity(self.N)
        self.Sinv = la.inv(self.S)
       
        print('   $Create Overlap:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')

    def CreateV (self):
        start = datetime.datetime.now()
        self.V = np.zeros((self.N, self.N))
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        R2 = (x1-x2)**2+(y1-y2)**2
        np.fill_diagonal(R2, 1)
        self.V = self.Hubbard/(np.sqrt(1+(1/self.Hubbard**2)*R2))
        np.fill_diagonal(self.V, self.Hubbard)
        
#        dx = np.max(x)-np.min(x)
#        dy = max(np.max(y)-np.min(y), dx/3)
#        self.X, self.Y = np.meshgrid(np.linspace(np.min(x)-0.2*dx, np.max(x)+0.2*dx, int(1.4*dx/0.1)), np.linspace(np.min(y)-0.2*dy, np.max(y)+0.2*dy, int(1.4*dy/0.1)))
#        self.Pot = np.zeros(np.shape(self.X))    
        
#        for i in range(self.N):
#            self.Pot += e0[Atom[i]]/(2*np.sqrt(1+0.6117*(((self.X-x[i])*5)**8+((self.Y-y[i])*5)**8)))
        print('   $Create V:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return self.V

class AGNR(Molecule):
    def __init__(self, N, L):
        super().__init__()
        
        self.Name = '%0.0f-AGNR(%0.0f)' % (N, L)
        
        for l in range(L):
            for n in range(N):
                self.AddAtom('C', x=1.5*l*dL+((n%2==0 and l%2==0) or (n%2==1 and l%2==1))*0.5*dL, y=n*np.sqrt(3)/2*dL, z=0)
            
class ZGNR(Molecule):
    def __init__(self, N, L):
        super().__init__()
        
        self.Name = '%0.0f-ZGNR(%0.0f)' % (N, L)
        
        for l in range(L):
            for n in range(2*N):
                if l%2==0 and (n%4==1 or n%4==2):
                    self.AddAtom('C', x=l/2*np.sqrt(3)*dL, y=np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
                elif l%2==1 and (n%4==0 or n%4==3):                    
                    self.AddAtom('C', x=l/2*np.sqrt(3)*dL, y=np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
             
class Junction_5_7_AGNR(Molecule):
    def __init__(self, NM):
        super().__init__()
        
        szPref = ''
        szTerm = ''
        
        X0=0
        alpha = 0
        for i, nm in enumerate(NM):
            Nlen = len(self.Atom)
            X0, nl1, nr1 = self.AddMonomer(nm[0], int(2*nm[1]), X0)
            rotPointEnd = np.array([np.max(self.Atom[np.arange(Nlen, self.N),1])+0.71, 3.67, 0])
            self.Parts.append(np.arange(Nlen, self.N))
            if i>0:
                if len(nm)==2:
                    alpha += 15*np.pi/360
                else:
                    alpha += nm[2]*np.pi/360
                    if nm[2]<0 and nm[3]=='up':                        
                        self.Translate(np.arange(Nlen, self.N), np.array([0, +np.sqrt(3)*1.42, 0]))
                    if nm[2]<0 and nm[3]=='down':                        
                        self.Translate(np.arange(Nlen, self.N), np.array([0, -np.sqrt(3)*1.42, 0]))
                        
                rotPointbeg = np.array([np.min(self.Atom[np.arange(Nlen, self.N),1])-0.71, 3.67, 0])
                rotPointEnd = np.array([np.max(self.Atom[np.arange(Nlen, self.N),1])+0.71, 3.67, 0])
                self.Translate(np.arange(Nlen, self.N), -rotPointbeg)
                self.Rotate(   np.arange(Nlen, self.N),    alpha)
                self.Translate(np.arange(Nlen, self.N), +rotPointBeg)
                
                rotPointEnd[0:2] = np.matmul([[np.cos(alpha), -np.sin(alpha)],[+np.sin(alpha), np.cos(alpha)]], rotPointEnd[0:2]-rotPointbeg[0:2])+rotPointBeg[0:2]

            rotPointBeg = rotPointEnd
            
            szPref = szPref +                         str(nm[0])
            szTerm = szTerm + (',' if i!=0 else '') + str(nm[1])
            
            
        self.Rotate(np.arange(0, self.N), -alpha/2)
        self.Translate(np.arange(0, self.N), -np.array([np.min(self.Atom[:,1]), np.min(self.Atom[:,2]), 0]))
        self.Name = szPref + '-GNR(' + szTerm + ')'
        self.UpdateMolecule(False)
        
    def AddMonomer(self, N, M, X0):        
        nl = len(self.Atom) + 4 + int(N==7)
        
        for m in range(M):
            for n in range(N):
                self.AddAtom('C', x=X0-(n%2-1)*np.sin(np.radians(30))*dL, y=n*np.cos(np.radians(30))*dL, z=0)
            X0+=(1+np.sin(np.radians(30)))*dL
            for n in range(N):
                self.AddAtom('C', x=X0+(n%2)*np.sin(np.radians(30))*dL, y=n*np.cos(np.radians(30))*dL, z=0)
            X0+=(1+np.sin(np.radians(30)))*dL
            
        nr = len(self.Atom) - 1 - int(N==7)
        return X0, nl, nr
                    
class CGNR(Molecule):
    def __init__(self, Nz, Na, Ez, Ea, Nm):
        super().__init__()
        
        self.Name = '%0.0f,%0.0f-%0.0f,%0.0f-CGNR(%0.0f)' % (Nz, Na, Ez, Ea, Nm)
        
        def AddMonomer(X0, Y0):
            for l in range(Nz):
                for n in range(2*Na):
                    if l%2==0 and (n%4==1 or n%4==2):
                        self.AddAtom('C', x=X0+l/2*np.sqrt(3)*dL, y=Y0+np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
                    elif l%2==1 and (n%4==0 or n%4==3):                    
                        self.AddAtom('C', x=X0+l/2*np.sqrt(3)*dL, y=Y0+np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
            
        dx = (2*Ez + Ea%2)*np.sqrt(3)/2*dL
        dy = Ea*3/2*dL
                        
        for m in range(Nm):
            AddMonomer(dx*m, dy*m)
            
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y, y)
        R2 = (x1-x2)**2+(y1-y2)**2

        atom_del = []
        for i in range(1,self.N):
            ind = np.argwhere(R2[i, 0:i-1]<dL/10)
            if len(ind)!=0:
                for j in ind[0]:
                    atom_del.append(j)
                    
        for i in atom_del[::-1]:
            self.UpdateAtom(i, 'DEL')
            
        self.Rotate(np.arange(0,self.N), -np.arctan(dy/dx))
        self.Translate(np.arange(0, self.N), [-np.min(self.Atom[:,1]), -np.min(self.Atom[:,2]), 0])
    
class AA_Corner_GNR(Molecule):
    def __init__(self, alpha, N1, N2, L1, L2):
        super().__init__()
        
        if alpha==60:
            self.CreateAA60(N1, N2, L1, L2)
        if alpha==120:            
            self.CreateAA120(N1, N2, L1, L2)
        self.Name = "%0.0f,%0.0f-AA-%0.0fCGNR(%0.0f,%0.0f)" % (N1, N2, alpha, L1, L1)
            
    def CreateAA60(self, N1, N2, L1, L2):
        x1 = -dL/2
        y1 = np.sqrt(3)/2*dL
        y2 = (N1*np.sqrt(3)/2-1/2)*dL
        x2 = 3/4*N2*dL+y2/np.sqrt(3)
        if N2>5:    x2 += dL/2
        if N2>8:    x2 += dL/2
        if N2>9:    x2 += dL/2
        if N2>11:   x2 += dL/2
        x3 = x2 + (L1*3/2-1/2)*dL 
        x4 = x2 + (L2*3/2-1)/2*dL
        y4 = y2 + (L2*3/2-1)*dL*np.sqrt(3)/2

        xtot = x3
        ytot = y4 + 1/2*N2*np.sqrt(3)/2*dL

        self.CreateArmchairSheet(int(ytot/np.sqrt(3)*2/dL)+2, int(xtot/3*2/dL)+2)
        
        #Carve out first part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>np.sqrt(3)*(x[i]-x1)+y1:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out second part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>y2 and y[i]<np.sqrt(3)*(x[i]-x2)+y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out third part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if x[i]>x3 and y[i]<y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out fourth part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>-1/np.sqrt(3)*(x[i]-x4)+y4 and y[i]>y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL')
            
#        self.AddAtom('N', x1, y1, 0)
#        self.AddAtom('N', x2, y2, 0)
#        self.AddAtom('N', x3, y2, 0)
#        self.AddAtom('N', x4, y4, 0)
        
        #Set contact points
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        for i in range(self.N):
            if x[i]>x3-dL and y[i]<y2:
                self.SetLead(i, 'Left')
        for i in range(self.N):
            if y[i]>-1/np.sqrt(3)*(x[i]-x4)+y4-dL and y[i]>y2:
                self.SetLead(i, 'Right')
                
        self.Rc = [x2, y2]
    
    def CreateAA120(self, N1, N2, L1, L2):
        y1 = -(N1-1)*np.sqrt(3)/2*dL-0.1
        x1 = (L1*3/2-1)*dL
        
        l2 = (L2*3/2)*dL
        x2 = x1+l2/2
        y2 = y1-l2*np.sqrt(3)/2
        
        l3 = N2*np.sqrt(3)/2*dL
        x3 = x2+l3*np.sqrt(3)/2
        y3 = y2+l3/2
        
        xtot = x3
        ytot = -y2
        
        self.CreateArmchairSheet(int(ytot/np.sqrt(3)*2/dL)+1, int(xtot*2/3/dL)+1)
        
        ymax = np.max(self.Atom[:,2])

        #Carve out first part
        y1 += ymax
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]<y1 and y[i]<np.sqrt(3)*(x1-x[i])+y1:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
        
        #Carve out second part
        y2 += ymax
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]<1/np.sqrt(3)*(x[i]-x2)+y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
        
        #Care out third part
        y3 += ymax
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>np.sqrt(3)*(x3-x[i])+y3:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
        
#        self.AddAtom('N', x1, y1, 0)
#        self.AddAtom('N', x2, y2, 0)
#        self.AddAtom('N', x3, y3, 0)#Set contact points
        
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        for i in range(self.N):
            if x[i]<dL/2:
                self.SetLead(i, 'Left')
        for i in range(self.N):
            if y[i]-dL/2<1/np.sqrt(3)*(x[i]-x2)+y2:
                self.SetLead(i, 'Right')
    
    def CreateArmchairSheet(self, N, L):
        for l in range(L):
            for n in range(N):
                self.AddAtom('C', x=1.5*l*dL+((n%2==0 and l%2==0) or (n%2==1 and l%2==1))*0.5*dL, y=n*np.sqrt(3)/2*dL, z=0)
 

class ZZ_Corner_GNR(Molecule):
    def __init__(self, alpha, N1, N2, L1, L2):
        super().__init__()
        
        if alpha==60:
            self.CreateZZ60(N1, N2, L1, L2)
        if alpha==120:            
            self.CreateZZ120(N1, N2, L1, L2)
                    
        self.Name = "%0.0f,%0.0f-ZZ-%0.0fCGNR(%0.0f,%0.0f)" % (N1, N2, alpha, L1, L1)
            
    def CreateZZ60(self, N1, N2, L1, L2):
        #Determine point for carving out
        x1 = -3/2*dL/np.sqrt(3)
        y1 = dL/2
        y2 = (N1*3/2-1/2)*dL
        x2 = (N2*3/2-1)*2/np.sqrt(3)+y2/np.sqrt(3)+dL
        if N2>4:    x2 += dL
        if N2>6:    x2 += dL
        x3 = x2 + np.sqrt(3)/2*L1*dL-0.1*dL
        x4 = x2 + (L2-1)*np.sqrt(3)/2*dL*1/2
        y4 = y2 + (L2-1)*np.sqrt(3)/2*dL*np.sqrt(3)/2+dL/2

        xtot = np.max((x3, x4))
        ytot = y4+(N2*3/2-1)/2*dL
        self.CreateZigzagSheet(int(ytot/dL*2/3)+2, int(xtot/dL*2/np.sqrt(3))+2)
        
        
        #Carve out first part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>np.sqrt(3)*(x[i]-x1)+y1:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out second part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>y2 and y[i]<np.sqrt(3)*(x[i]-x2)+y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out third part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if x[i]>x3 and y[i]<y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out fourth part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        del_atom = []
        for i in range(self.N):
            if y[i]>-1/np.sqrt(3)*(x[i]-x4)+y4 and y[i]>y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
        
#        self.AddAtom('N', x1, y1, 0)
#        self.AddAtom('N', x2, y2, 0)
#        self.AddAtom('N', x4, y4, 0)

        #Set contact points
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        for i in range(self.N):
            if x[i]>x3-dL/2*np.sqrt(3) and y[i]<y2:
                self.SetLead(i, 'Left')
        for i in range(self.N):
            if y[i]>-1/np.sqrt(3)*(x[i]-x4)+y4-dL and y[i]>y2:
                self.SetLead(i, 'Right')
            
    def CreateZZ120(self, N1, N2, L1, L2):
        Ntot = int(N1+(L2-1)/2)+2
        Ntot += (Ntot-N1)%2
        Ltot = int(L1+L2/2-1/2+3/2*N2-1-1/2*N2%2)+2
        self.CreateZigzagSheet(Ntot, Ltot)
            
        #Carve out first part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        y1 = np.max(y)-(3/2*N1-1/2)*dL
        x1 = (L1-1)*np.sqrt(3)/2*dL
        del_atom = []
        for i in range(self.N):
            if y[i]<y1 and y[i]<np.sqrt(3)*(x1-x[i])+y1:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
                
        #Carve out second part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        y2 = np.max(y)+dL/2
        x2 = x1 + (y2-y1)/np.sqrt(3)
        del_atom = []
        for i in range(self.N):
            if y[i]>np.sqrt(3)*(x2-x[i])+y2:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Carve out third part
        x = self.Atom[:,1]
        y = self.Atom[:,2]
        y3 = y1-(L2-1)/2*3/2*dL
        x3 = x1-(y3-y1)/np.sqrt(3)
        
        del_atom = []
        for i in range(self.N):
            if y[i]<0.57*(x[i]-x3)+y3:
                del_atom.append(i)
        for i in del_atom[::-1]:
            self.UpdateAtom(i, 'DEL') 
            
        #Set leads
        for i in np.argwhere(self.Atom[:,1]<dL/2)[:,0]:
            self.SetLead(i, 'Left')            
#        self.Rotate(range(self.N), np.radians(60))
        for i in np.argwhere(self.Atom[:,1]>np.max(self.Atom[:,1])-dL/2)[:,0]:
            self.SetLead(i, 'Right')
#        self.Rotate(range(self.N), np.radians(-30))
        
#        self.AddAtom('N', x1, y1, 0)
#        self.AddAtom('N', x2, y2, 0)
#        self.AddAtom('N', x3, y3, 0)
        
        
            
    def CreateZigzagSheet(self, N, L):
        for l in range(L):
            for n in range(2*N):
                if l%2==0 and (n%4==1 or n%4==2):
                    self.AddAtom('C', x=l/2*np.sqrt(3)*dL, y=np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
                elif l%2==1 and (n%4==0 or n%4==3):                    
                    self.AddAtom('C', x=l/2*np.sqrt(3)*dL, y=np.floor(n/2)*1.5*dL+n%2*0.5*dL, z=0)
        
class AZ_Corner_GNR(Molecule):
    def __init__(self, Na, Nz, La, Lz):
        super().__init__()
        self.Name = "%0.0f,%0.0f-AZ-90CGNR(%0.0f,%0.0f)" % (Na, Nz, La, Lz)
        
        for l in range(La+Nz):
            for n in range(Na+Lz):
                self.AddAtom('C', x=1.5*l*dL+((n%2==0 and l%2==0) or (n%2==1 and l%2==1))*0.5*dL, y=n*np.sqrt(3)/2*dL, z=0)
       
        x = self.Atom[:,1]
        y = self.Atom[:,2]

        x0 = (Nz*3/2-1+1/4)*dL
        y0 = ((Na-1)*np.sqrt(3)/2+1/2)*dL
        for i in np.argwhere(np.logical_and(x>x0, y>y0))[::-1]:
            self.UpdateAtom(i, 'DEL')         
            
        x = self.Atom[:,1]
        y = self.Atom[:,2]

        for i in np.argwhere(x>=np.max(x)-dL/4)[:,0]:
            self.SetLead(i, 'Left')
        for i in np.argwhere(y>=np.max(y)-dL/2)[:,0]:
            self.SetLead(i, 'Right')
            
#        self.Rotate(np.arange(self.N), np.radians(-135))
        self.UpdateMolecule()  
        
        
if __name__ == '__main__':
    from AtomParam import cAtom
    N = 9
    L = 4*int(np.sqrt(3)/3*N)
    L += L%2
    
    Mol1 = AA_Corner_GNR(60, 13, 13, 11, 11)
#    Mol1.SetLeads([-1, -1])
    Mol1.SetGam_b(0.000001)
    Mol1.UpdateMolecule()
    Graph = Graphics()
    Graph.SetMolecule(Mol1)
#    Mol1.SetOutputFolder(Mol1.Name + "/")
#    from Magnetics import *
    
#    Mag = Magnetics(Mol1)
#    Mag.PerformAnalysis()

    plt.show()
