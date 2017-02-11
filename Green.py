"""Import necessary package"""
import numpy as np
from numpy import sqrt
import numpy.linalg as la
from copy import copy, deepcopy
import math as math
import datetime
import scipy.linalg as scLA
import matplotlib.pyplot as plt

"""Import Custom package"""
from AtomParam import *
from Graphics import *

class Green(object):
    """ Represents a Molecule including hamiltonian and transmission and current calculations 
        Attributes
        HamExt                      Hamiltonian of the Molecule, with interaction with leads
        N                           Size of the Hamiltonian
        Gleft, Gright               Interaction strengths with the leads
        gam_l, gam_r                Gamma matrices
        eigval, eigvec              Respectively real and imaginary part of the eigenvalues of the Hamiltonian
        C                           Matrix with the eigenvectors
        Gates                       Array of gate voltages which have to be calculated
        T,P,I                       Transmission amplitude and phase and current

        Methods
        InitGreen()                 Sets the initial parameters for the interactino with the leads
        SetG(Gleft, Gright)         Set the interaction strengths with the leads
        SetOverlap()
        CreateHamExt(Bias, scType, delta, Beta)             Creates the Hamiltonian that includes the interaction with the leads in different consistency ways (no consistency, Hubbard, Coulomb)
        GetHamExt(D, scType)
        CalcEigValVec()     
        UpdateGates(Gmin,Gmax)      Sets the range of the gate voltages betweeen the lowest and heighest eigenenergie plus 5 times gamma
        CalcGamma()                 Calculates the gamma matrices in the basis of the HamExt
        CalcGe()
        CalcBetaMat()               Calculates the Beta matrices
        
        Green(E)                    Calculates the Green matrix
        GreenDiag(E)
        GreenInv (E)

        Density(Bias)               Calculates the Density matrix, with an upper and lower energy boundary
        DensityDiscrete(Bias)
        DensityAnalytic(Bias)

        CalcDOS()

        TransmissionMatrix(E)       Calculates the complete transmission matrix

        Transmission()              Calculates the transmission based on one of the methods below
        TransmissionFromSeldenthuis Calculates the transmission based on the expression from Seldenthuis
        TransmissionFromMatrix()    Calculates the transmission based on the trace of the transmission matrix
        TransmissionFromOrbitals()  Calculates the transmission based on the individual transmissions from the orbitals

        TransmissionOrbital(iOrbit) Calculates the transmission of an orbital based on one of the methods below
        TransmissionOrbitalFromSeldenthuis(iOrbit) Calculates the transmission of an orbital based on the method described by Seldenthuis
        
        Current(bias)               Calculates the current based on one of the methods below
        CurrentFromSeldenthuis(bias)Calculates the current based on the expression from Seldenthuis
        CurrentFromMatrixDiscrete(bias) Calculates the current from the trace of the discretely obtained current matrix
        CurrentFromMatrixAnalytic(bias) Calculates the current from the trace of the analytically obtained current matrix

        CurrentMatrix(bias, E)      Calculates the current matrix based on one of the methods below
        CurrentMatrixDiscrete(bias, E, Nstep) Calculates the current matrix based on a Riemman sum of the transmission matrices
        CurrentMatrixAnalytic(bias, E)        Calculates the current matrix based on an analytic expression

        LocalCurrents(bias, E)      Calculates the local currents based on an expression by Solomon
        """
    def __init__(self):
        self.Gam_L = Gam_L
        self.Gam_R = Gam_R
        self.Gam_S = Gam_S        

        self.lpL = []
        self.lpR = []
        self.lpS = np.zeros((self.N))        
        
        self.Gam_b = Gam_b
        self.Temp = 0
        
        self.bOverlap = True
        self.bSpin = False
        self.Consistency = 'Not self consistent'
        self.Hubbard = Hubbard

        self.Efermi = E0
        self.Bias = .01
        
        print('Initial parameters: \n', 'Gleft', self.Gam_L, '\n Gright', self.Gam_L, '\n Overlap', self.bOverlap, '\n Spin', self.bSpin)

    def UpdateMolecule(self, bNext=True):
        self.CreateHam()
        self.CreateOverlap()
        self.CreateV()
        return self.UpdateSystem() if bNext else True
        
    def SetFermi(self, Efermi=None):
        if Efermi==None:
            self.Efermi=self.GetFermi()
        self.Efermi = Efermi
        
    def SetGam_LR (self, gam_l=Gam_L, gam_r=Gam_R, gam_s=Gam_S):
        self.Gam_L = gam_l
        self.Gam_R = gam_r
        self.Gam_S = gam_s
        print('Gamma set to', self.Gam_L, '(L)', self.Gam_R, '(R)', self.Gam_S, '(S)')
        return self.UpdateSystem()

    def SetGam_b(self, Gam_b):
        self.Gam_b = Gam_b
        print('Gam_b set to:', self.Gam_b)
        return self.UpdateMolecule()
            
    def SetLead (self, iAtom, side):
        if side=='Left':
            self.lpL.append(iAtom)
        elif side=='Right':
            self.lpR.append(iAtom)
        elif side=='STM':
            self.lpL = iAtom
        else:
            pass
            
    def SetLeads (self, lim=(0,0)):
        self.lpL = []
        self.lpR = []
        
        x = self.Atom[:,1]
        
        xmin = min(x)
        xmax = max(x)

        for i in range(len(x)):
            if x[i]-xmin<=lim[0]:
                self.SetLead(i, 'Left')
            if xmax-x[i]<=lim[1]:
                self.SetLead(i, 'Right')
                
    def SetSurface(self, z=None):
        if z==None:
            self.lpS = np.zeros(len(self.N))
        else:
            self.lpS = np.exp(-z/1.5)        

    def SetOverlap(self, bOverlap=False):
        self.bOverlap=bOverlap
        self.CreateOverlap()
        print("Overlap set to:", self.bOverlap)
        return self.UpdateSystem()
        
    def SetSpin (self, bSpin=False):
        self.bSpin = bSpin
        print ("Spin set to", self.bSpin)
        return self.UpdateSystem()
        
    def SetConsistency(self, consistency):
        self.Consistency = consistency
        print ("Consistency set to:", self.Consistency)
        return self.UpdateSystem()
        
    def SetBias(self, bias):
        self.Bias = bias
        print ("Bias voltage set to: ", self.Bias, "V")
        return self.UpdateSystem
        
    def UpdateSystem(self, bNext=True):
        self.CreateHamExt ()
        self.CalcGamma()
        return self.UpdateQuantities() if bNext else True
        
    def UpdateQuantities(self):
        self.GetDOS()
        self.Density()
        
        self.Transmission()
        self.Current()
        return True
        
        
        
            
    def UpdateGates (self, Gmin=None, Gmax=None):#-4, Gmax=+4):
        if Gmin==None:
            Gmin = np.amin(self.e_arr if self.bSpin else self.e_arr[0])-5*np.amax(abs(self.gam))
        if Gmax==None:
            Gmax = np.amax(self.e_arr if self.bSpin else self.e_arr[0])+5*np.amax(abs(self.gam))
            
        self.Gates = np.arange(np.real(Gmin), np.real(Gmax), float(np.real(Gmax-Gmin))/1000)
        return self.Gates
        
        
    def GetFermi(self, charge=0):
        Nelectrons = self.N-charge
        e = np.concatenate((self.e_arr[0], self.e_arr[1] if self.bSpin else self.e_arr[0]))
        e = np.sort(e)
        return np.real(e[Nelectrons-1]+e[Nelectrons])/2
        
    def CreateHamExt(self, delta=10**(-5), Beta=0.5, bPlot=False):
        self.CreateHamExtNew(delta, Beta, bPlot)
        
    def CreateHamExtNew(self, delta=10**(-7), Beta=0.5, bPlot=False):
        
        """Create necessary data structures"""
        Nmax = 100
        Efermi1 = []
        rho1    = []
        rho1std = []
        rho1Atom = []
        rho1AtomMax = []  
        Efermi2 = []
        rho2 = []   
        rho2std = []
        rho2Atom = []
        rho2AtomMax = []   
        self.HamExt = np.zeros((2, self.N, self.N), dtype=complex)
        self.HamExt[0] = self.Ham 
        self.HamExt[1] = self.Ham
        
        """Create function that calculates electron-electron potential"""        
        def Uelectron (Rho):
            Rho = Rho if self.bSpin else np.array([Rho, Rho])
            U = np.zeros((2, self.N))
        
            #Hubbard electron-electron interaction
            if self.bSpin:
                U[0] = (Rho[1]-1/2)*np.diag(self.V)
                U[1] = (Rho[0]-1/2)*np.diag(self.V)
            else:
                U[0] = (Rho[0]-1/2)*np.diag(self.V)
                
            #Pariser-Par-Pople electron-electron interaction
            if self.Consistency=='PPP':
                for i in range (self.N):
                    for j in range(self.N):
                        if not i==j:
                            U[:,i] += (np.sum(Rho[:,j])-1)*self.V[i,j]
            return U

#        self.CalcEigValVec()        
#        self.CalcGamma()
#        self.GetDOS()
#        self.Density()
        
#        graph = Graphics()
#        graph.SetMolecule(self)       
        
        """Stage 1: Solve isolated system self-consistently"""        
        if self.Consistency!='Not self consistent': 
            for Hub in [1,2,3]:
                self.Hubbard = Hub
                self.CreateV()
                for i in range(Nmax):
                    print("Stage 1: Iteration", i)
                    self.CalcEigValVec()
                    self.Efermi = self.GetFermi()
                
                    #Keep track of variables
                    Efermi1.append(self.Efermi)
                    try:
                        rho_prev = copy(rho) 
                    except:
                        rho_prev = self.Density() if self.bSpin else self.Density()[0]
                    rho = self.Density() if self.bSpin else self.Density()[0]
                    rho = rho/np.mean(rho)*0.5  #Normalize, should actually not be necessary
                    rho1.append(np.mean(rho))
                    rho1std.append(np.std(rho))
                    rho1Atom.append(np.mean(abs(rho-rho_prev)))
                    rho1AtomMax.append(np.max(abs(rho-rho_prev)))
                
                    print(self.Efermi)
                    self.GetDOS()
#                   graph.UpdateSystem()
#                   graph.Save('Test/' + self.Name + 'Iteration ' + str(i) + '.png')
                
                    if rho1AtomMax[-1]<delta and i>0: break
                    U = Uelectron((1-Beta)*rho + Beta*rho_prev)
                
                    #Apply assymmetry for spin
                    if self.bSpin and i==0:
                        print('Applying the assymmetry')
                        U[0] +=0.1
                        U[1] -=0.1
#                       U[0, np.argwhere(self.Atom[:,2]==np.min(self.Atom[:,2]))] += 10
#                       U[0, np.argwhere(self.Atom[:,2]==np.max(self.Atom[:,2]))] += 10
#                       U[1, np.argwhere(self.Atom[:,2]==np.min(self.Atom[:,2]))] -= 1
#                       U[0, np.argwhere(self.Atom[:,2]==np.max(self.Atom[:,2]))] += 0.1
#                       U[1, np.argwhere(self.Atom[:,2]==np.max(self.Atom[:,2]))] -= 0.1
#                   print(U)
                
                    for spin in range(1+self.bSpin):
                        self.HamExt[spin] = self.Ham + np.diagflat(U[spin])
        else:
            self.CalcEigValVec()
#            self.Efermi = self.GetFermi()
            
        """Stage 2: Solve total system self-consistently"""
        #Add lead self-energies
        for spin in range(self.bSpin+1):
            if len(self.lpL)!=0:    self.HamExt[spin][self.lpL, self.lpL] -= 0.5j*self.Gam_L
            if len(self.lpR)!=0:    self.HamExt[spin][self.lpR, self.lpR] -= 0.5j*self.Gam_R  
            if len(self.lpS)!=0:    self.HamExt[spin][range(self.N),range(self.N)] -= 0.5j*self.Gam_S*self.lpS.reshape(self.N)
            
        if self.Consistency!='Not self consistent':       
            HamExtOrg = deepcopy(self.HamExt-np.diagflat(U[spin]))   
        
            for i in range(Nmax):
                print("Stage 2: Iteration", i)
                self.CalcEigValVec()
#                self.Efermi = self.GetFermi()
                
                #Keep track of variables
                Efermi2.append(self.Efermi)
                rho_prev = copy(rho)
                rho = self.Density() if self.bSpin else self.Density()[0]
                rho = rho/np.mean(rho)*0.5  #Normalize, should actually not be necessary
                rho2.append(np.mean(rho))
                rho2std.append(np.std(rho))
                rho2Atom.append(np.mean(abs(rho-rho_prev)))
                rho2AtomMax.append(np.max(abs(rho-rho_prev)))
                if rho2AtomMax[-1]<delta and i>0: break
                U =  Uelectron((1-Beta)*rho + Beta*rho_prev)
#                print("U", U)
                for spin in range(1+self.bSpin):
                    self.HamExt[0] = HamExtOrg[0] + np.diagflat(U[0])
        else:
            self.CalcEigValVec()
        
#        print(rho1Atom, rho2Atom, rho1AtomMax, rho2AtomMax)
        
        if bPlot:
            x1 = np.arange(len(rho1))
            x2 = np.arange(len(rho2))+ len(rho1)
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1)
            ax.plot(x1, Efermi1, 'b')
            ax.plot(x2, Efermi2, 'lightblue')
            ax.set_ylabel('$\epsilon_F$')
            ax.set_xlabel('$Iteration$')
            ax = ax.twinx()
            ax.errorbar(x1, rho1, rho1std, c='g')
            ax.errorbar(x2, rho2, rho2std, c='limegreen')
            ax.set_ylabel(r'$\rho$')
#            ax = ax.twinx()
#            ax.plot(x1, mulfac, 'k')
            ax = fig.add_subplot(1,2,2)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax = ax.twinx()
            try:    ax.semilogy(x1, rho1Atom, 'mediumorchid')
            except: pass
            try:    ax.semilogy(x2, rho2Atom, 'plum')
            except: pass
            try:    ax.semilogy(x1, rho1AtomMax, 'r')
            except: pass
            try:    ax.semilogy(x2, rho2AtomMax, 'firebrick')
            except: pass
            ax.semilogy(np.concatenate((x1, x2)), np.ones(len(x1)+len(x2))*delta, 'k:')
            ax.set_ylabel(r'$\left| {{\rho ^n} - {\rho ^{n - 1}}} \right|$')
            ax.set_xlabel('$Iteration$')
        
#        
#    def CreatePotential(self):
#        start = datetime.datetime.now()
#        self.PotExt = deepcopy(self.Pot)*0
#
#        Atom, x, y, z = self.GetCoord()      
#        Dn = np.sum(self.Density(0),axis=0)
#        for i in range(self.N):
#            self.PotExt += Hubbard/(2*np.sqrt(1+0.6117*(((self.X-x[i]))**2+((self.Y-y[i]))**2)))*(Dn[i]-0.5-0.5*self.bSpin)
#        print('   $Create Potential', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
##        fig, ax = plt.subplots(1, 1)
##        cax2 = ax.pcolormesh(self.X, self.Y, self.Pot)
##        cbax2 = fig.add_axes([0.40, 0.05, 0.25, 0.03]) 
##        plt.colorbar(cax2, orientation='horizontal', cax=cbax2)
##        ax.axis('equal')
##        plt.show()
                   
    def CalcEigValVec(self):
        start = datetime.datetime.now()
        """ Calculate eigenvalues, normalized eigenvectors in order to diagonalize Green matrix """
        self.eigvec    = np.zeros((2, self.N, self.N),dtype=complex)
        self.eigvecinv = np.zeros((2, self.N, self.N),dtype=complex)
        self.eigval    = np.zeros((2, self.N),        dtype=complex)
        self.e_arr     = np.zeros((2, self.N),        )
        self.gam       = np.zeros((2, self.N),        )
        
        for iSpin in range(1+self.bSpin):
            if self.bOverlap:
                self.eigval[iSpin],self.eigvec[iSpin] = scLA.eig(self.HamExt[iSpin], self.S)    
            else:
                self.eigval[iSpin],self.eigvec[iSpin] = scLA.eig(self.HamExt[iSpin])
                
            self.eigvec[iSpin] /= np.swapaxes(np.sqrt(np.sum(self.eigvec[iSpin]*np.conjugate(self.eigvec[iSpin]), axis=1))*np.ones((self.N, self.N)), 0, 1)  
                
            idx = np.real(self.eigval[iSpin]).argsort()
            self.eigval[iSpin] = self.eigval[iSpin][idx]
            self.eigvec[iSpin] = self.eigvec[iSpin][:,idx]     
            
            self.eigvecinv[iSpin] = la.inv(self.eigvec[iSpin])
            
            self.e_arr[iSpin] =  np.real(self.eigval[iSpin])

            self.e_arr[iSpin] =  np.real(self.eigval[iSpin])
            self.gam[iSpin]   = -np.imag(self.eigval[iSpin])
        self.UpdateGates()

        print('   $CalcEigValVec - Overlap', self.bOverlap, int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return

    def CalcGamma(self):
        start = datetime.datetime.now()
        """Gamma Matrices, modeling of the interaction with both leads"""
        gmat_l, gmat_r = np.zeros((2, self.N, self.N), dtype=complex) 
        
        gmat_l[self.lpL, self.lpL] += self.Gam_L
        if len(self.lpS)!=0:    gmat_r[range(self.N),range(self.N)] += self.Gam_S*self.lpS.reshape(self.N)
        gmat_r[self.lpR, self.lpR] += self.Gam_R
               
        self.CalcEigValVec()

        """ Diagonalize Gamma matrices into basis of HamExt"""
        self.gam_l, self.gam_r, self.gam_l_t, self.gam_r_t = np.zeros((4,2,self.N, self.N), dtype=complex)
        for spin in range(1+self.bSpin):
            self.gam_l[spin]   = la.multi_dot([np.transpose(self.eigvec[spin]), gmat_l, np.conjugate(self.eigvec[spin])])
            self.gam_r[spin]   = la.multi_dot([np.transpose(self.eigvec[spin]), gmat_r, np.conjugate(self.eigvec[spin])])
            self.gam_l_t[spin] = la.multi_dot([self.eigvecinv[spin], self.Sinv, gmat_l, np.conjugate(np.transpose(self.Sinv)), np.conjugate(np.transpose(self.eigvecinv[spin]))])
            self.gam_r_t[spin] = la.multi_dot([self.eigvecinv[spin], self.Sinv, gmat_r, np.transpose(np.conjugate(self.Sinv)), np.transpose(np.conjugate(self.eigvecinv[spin]))])
        
        print('   $Calc Gamma:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return
          
    
        """-----------------------------------Green----------------------------------------"""
    def Green (self, e, Spin=0):
        return self.GreenInv (e, Spin)   #Atom basis

    def GreenDiag(self, e, Spin=0):
#        start = datetime.datetime.now()
        G = la.multi_dot([self.eigvec[Spin], np.diag(1.0/(e-self.eigval[Spin])), self.eigvecinv[Spin], self.Sinv ])
#        print('   $Green Diag:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return G    #Atom basis

    def GreenInv (self, e, Spin=0):
#        start = datetime.datetime.now()
        G =  la.inv(e*self.S-self.HamExt[Spin])    
#        print('   $Green Inv:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return G     #Atom basis
            
        """------------------------------------DOS-----------------------------------------"""
    def GetDOS (self):
        return self.DOSAnalytical()
        
    def DOSAnalytical(self):
        start = datetime.datetime.now()
        self.DOS = np.zeros((2, len(self.Gates)))
        
        for spin in range(self.bSpin+1):
            numerator = np.sum(self.eigvec[spin]*np.transpose(np.dot(self.eigvecinv[spin], self.Sinv)), axis=0)
            numerator, ee = np.meshgrid(numerator, self.Gates)
            ei, ee = np.meshgrid(self.eigval[spin], self.Gates)
            self.DOS[spin]=-1/np.pi*np.imag(np.sum(numerator/(ee-ei), axis=1))
        print('   $DOS Analytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return self.DOS
        
    def DOSGreen(self):
        start = datetime.datetime.now()
        self.DOS = np.zeros((2, len(self.Gates)))
        
        for spin in range(self.bSpin+1):
            for i, ei in enumerate(self.Gates):
                self.DOS[spin][i] = -1/math.pi*np.trace(np.imag(self.Green(Spin=spin, e=ei)))
        
        print('   $DOS Green:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return self.DOS
        
#    def PartDOS(self, atoms):
#        start = datetime.datetime.now()
#        PartDOS = np.zeros((2, len(self.Gates)))
#        
#        for spin in range(self.bSpin+1):
#            numerator = np.sum(self.eigvec[spin]*np.transpose(np.dot(self.eigvecinv[spin], self.Sinv)), axis=0)
#            numerator, ee = np.meshgrid(numerator, self.Gates)
#            ei, ee = np.meshgrid(self.eigval[spin], self.Gates)
#            print(ei.shape)
#            PartDOS[spin]=-1/np.pi*np.imag(np.sum((numerator/(ee+1j*self.Gam_b-ei))[:,atoms], axis=1))
#            
#        print('   $Part DOS:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
#        return PartDOS
    
        """-----------------------------------LDOS-----------------------------------------"""    
    def LDOS(self, e):
        return self.LDOSAnalytical(e)
        
    def LDOSAnalytical(self,e):
        start = datetime.datetime.now()
            
        LDOS = np.zeros((2, self.N))
        
        for spin in range(self.bSpin+1):
            print('Obtain spin', spin)
            part2 = e-self.eigval[spin]
            part2, dummy = np.meshgrid(part2, np.zeros(self.N))
            LDOS[spin] = -1/np.pi*np.imag(np.sum(self.eigvec[spin]*np.transpose(np.dot(self.eigvecinv[spin],self.Sinv))/part2, axis=1))

        print('   $LDOS Analytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return LDOS
        
    def LDOSGreen(self,e, bTemp=True):
        start = datetime.datetime.now()
        
        LDOS = np.zeros((2, self.N))
        for spin in range(self.bSpin+1):
            G = self.GreenInv(Spin=spin, e=e)
            LDOS[spin] = -1/math.pi*np.diag(np.imag(G))
        
        print('   $LDOS Green:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return LDOS
        
        """----------------------------------Density---------------------------------------"""
    def Density(self, Spin=-1):
        start = datetime.datetime.now()
        if Spin==-1:
            self.LD = np.zeros((2, self.N))
            for spin in range(self.bSpin+1):
                self.LD[spin] = self.DensityAnalytical(Spin=spin)
            print('   $Density:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
            return self.LD
        else:
            LD = self.DensityAnalytical(Spin=Spin)
            self.LD[Spin] = LD
            print('   $Density:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
            return LD
        
    def DensityAnalytical(self, Spin, Nsteps=1000):
        start = datetime.datetime.now()
        kB = 8.6*10**(-5)
        LD = np.zeros((self.N))
        
#        e = np.linspace(start=np.min(self.e_arr)-250*(np.max(abs(self.gam))), stop=self.Efermi+5*(kB*self.Temp), num=Nsteps, dtype=np.float32)
        E = LogLinLogSpace([ np.min(self.e_arr)-1000*(np.max(abs(self.gam))), np.min(self.e_arr)-100*(np.max(abs(self.gam))), np.max(self.e_arr)+100*(np.max(abs(self.gam))), np.max(self.e_arr)+1000*(np.max(abs(self.gam)))], N=Nsteps) #self.Efermi+5*(kB*self.Temp), self.Efermi+50*(kB*self.Temp)], N=10000)        
        
        ei, ee = np.meshgrid(self.eigval[Spin], (E[1:]+E[0:-1])/2  )
        ei, de = np.meshgrid(self.eigval[Spin], (E[1:]-E[0:-1]))
        
        if self.Temp==0:    part2 = np.sum(de/(ee-ei), axis=0)
        else:               part2 = np.sum(de/(ee-ei)*1/(np.exp((ee-self.Efermi)/(kB*self.Temp))+1), axis=0)
        
        part2, d = np.meshgrid(part2, np.zeros(self.N))
    
        LD = -1/np.pi*np.imag(np.sum(self.eigvec[Spin]*np.transpose(np.dot(self.eigvecinv[Spin], self.Sinv))*part2, axis=1))
        print('   $Density Analytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return LD    
        
    def DensityGreen(self, Spin, Nsteps=1000):
        start = datetime.datetime.now()
        LD = np.zeros((self.N))
        E = LogLinLogSpace([ np.min(self.e_arr)-1000*(np.max(abs(self.gam))), np.min(self.e_arr)-100*(np.max(abs(self.gam))), np.max(self.e_arr)+100*(np.max(abs(self.gam))), np.max(self.e_arr)+1000*(np.max(abs(self.gam)))], N=Nsteps) #self.Efermi+5*(kB*self.Temp), self.Efermi+50*(kB*self.Temp)], N=10000)        
        de = (E[1:]-E[0:-1])
        E = (E[1:]+E[0:-1])/2
        for i, ei in enumerate(E):
            LD += -1/math.pi*np.diag(np.imag(self.Green(ei, Spin)))*de[i]
        print('   $Density Green:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return LD  #Atom basis
        
                        
        """--------------------------------Transmission------------------------------------"""
    def Transmission (self):
        self.T = np.real(self.TransmissionAnalytical (self.Gates))                     # Only for From Seldenthuis, From Matrix
        return self.T            

    def TransmissionAnalytical (self, e):
        start = datetime.datetime.now()
        
        """Calculate transmission"""
        T = np.zeros((2, len(e)), dtype=complex)
        for Spin in range(1+self.bSpin):
            for i in range(self.N):
                for j in range(self.N):
                    T[Spin] += self.gam_l[Spin][i,j]*self.gam_r_t[Spin][i,j]/((e - self.eigval[Spin][i])*(e - np.conjugate(self.eigval[Spin][j])))
                    
        print('   $TransmissionAnalytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return T
        
    def TransmissionGreen(self, e):
#        start = datetime.datetime.now()
        
        """Gamma Matrices, modeling of the interaction with both leads"""
        gmat_l, gmat_r = np.zeros((2, self.N, self.N), dtype=complex) 
        
        gmat_l[self.lpL, self.lpL] += self.Gam_L
        if len(self.lpS)!=0:    gmat_r[range(self.N),range(self.N)] += self.Gam_S*self.lpS.reshape(self.N)
        gmat_r[self.lpR, self.lpR] += self.Gam_R
        
        """Calculate transmission"""
        T = np.zeros((2,len(e)))
        for Spin in range(1+self.bSpin):
            for i, ei in enumerate(e):
                G = self.Green(ei)
                T[Spin, i] = np.trace(la.multi_dot([G, gmat_l, np.transpose(np.conjugate(G)), gmat_r]))
#        print('   $TransmissionGreen:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return T           

        """----------------------------------Current---------------------------------------"""
    def Current (self):
        self.I = self.CurrentAnalytical(mu_L=self.Gates+self.Bias, mu_R = self.Gates)
        return self.I
    
    def CurrentAnalytical (self, mu_L, mu_R):
        start = datetime.datetime.now()
                
        """Calculate current"""
        I = np.zeros((2, len(mu_L)), dtype=complex)
        for Spin in range(self.bSpin+1):
            log_term_plus  = np.zeros((self.N,len(mu_L)), dtype=complex)
            log_term_minus = np.zeros((self.N,len(mu_L)), dtype=complex)
            for i in range(self.N):
                log_term_plus [i,:] = np.log((mu_R-             self.eigval[Spin][i]) /(mu_L-             self.eigval[Spin][i]))
                log_term_minus[i,:] = np.log((mu_R-np.conjugate(self.eigval[Spin][i]))/(mu_L-np.conjugate(self.eigval[Spin][i])))
            for i in range(self.N):
                for j in range(self.N):
                    I[Spin] += self.gam_l[Spin][i,j]*self.gam_r_t[Spin][i,j]/(self.eigval[Spin][i]-np.conjugate(self.eigval[Spin][j]))*(log_term_plus[i,:]-log_term_minus[j,:])

        I *= 1/(2*np.pi)*0.00662361795*10**9
        I *= (2-self.bSpin)
        
        print('   $CurrentAnalytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return I
        
    def CurrentGreen(self, mu_L, mu_R, N_steps=100):
        start = datetime.datetime.now()
        
        I = np.zeros((2, len(mu_L)), dtype=complex) 
        for mu in range(len(mu_L)):
            de = mu_L[mu]-mu_R[mu]
            for ei in np.linspace(mu_R[mu], mu_L[mu], N_steps):
                I[:, mu] += self.TransmissionGreen([ei])[0]*de
                
        I *= 1/(2*np.pi)*0.00662361795*10**9
        I *= (2-self.bSpin)
            
        print('   $CurrentAnalytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return I
        
        """-------------------------------Local Transmission------------------------------------"""
    def LocalTransmission(self, e):
        T = np.zeros((2, 2, self.N, self.N))
        for Spin in range(1+self.bSpin):
            ei, ej = np.meshgrid(1/(e-np.conjugate(self.eigval[Spin])), 1/(e-self.eigval[Spin]))
            T[Spin] = self.LocalTransmissionAnalytical([ei*ej, np.zeros((self.N, self.N))], Spin)
#            T[Spin] = self.LocalTransmissionGreen(e, [True, False], Spin)
        
        if not self.bSpin:
            T[1] = T[0]
            
        #Check for non-zero net transmission:        
        for i in range(self.N):
            In = np.sum(T[:,:,i,:])/ np.sum(abs(T[:,:,i,:]))
            if abs(In)>10**(-5):
                szLead = 'and not connected!' if i not in self.lpL and i not in self.lpR else 'L' if i in self.lpL else 'R'
                print('Non-zero net current for', i, '(', int(10000*In)/100.0, '%)', szLead)
        return T        
            
    def LocalTransmissionGreen(self, e, bLeads, Spin=0):
        start = datetime.datetime.now()
        
        V = deepcopy(self.Ham)
        np.fill_diagonal(V, np.zeros(self.N))
        
        Gret = self.GreenInv(e, Spin=Spin)        
        Gadv = np.transpose(np.conjugate(Gret))
            
        Gam_L = np.zeros((self.N, self.N))
        Gam_L[self.lpL, self.lpL] += self.Gam_L
        Gam_R = np.zeros((self.N, self.N))
        if len(self.lpS)!=0:    Gam_R[range(self.N),range(self.N)] += self.Gam_S*self.lpS.reshape(self.N)
        Gam_R[self.lpR, self.lpR] += self.Gam_R
            
        
        T = np.zeros((2, self.N, self.N), dtype=complex)
        
        if bLeads[0]:
            GGG = la.multi_dot([Gret, Gam_L, Gadv])
            T[0] += 1j*(V*(np.transpose(GGG)-GGG))
        if bLeads[1]:
            GGG = la.multi_dot([Gret, Gam_R, Gadv])
            T[1] -= 1j*(V*(GGG-np.transpose(GGG)))
                
        print('   $Local Currents Green:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return -T
        
    def LocalTransmissionAnalytical(self, Ups, Spin=0):
        start = datetime.datetime.now()
        
        tau = deepcopy(self.Ham)
        np.fill_diagonal(tau, np.zeros(self.N))       
                     
        GGG = la.multi_dot([self.eigvec[Spin], self.gam_l_t[Spin]*Ups[0], np.transpose(np.conjugate(self.eigvec[Spin]))])
        T_l = 1j*(tau*(GGG-np.transpose(GGG)))
        GGG = la.multi_dot([self.eigvec[Spin], self.gam_r_t[Spin]*Ups[1], np.transpose(np.conjugate(self.eigvec[Spin]))])
        T_r = 1j*(tau*(GGG-np.transpose(GGG)))
 
        print('   $Local Currents Analytical:', int((datetime.datetime.now()-start).total_seconds()*1000), 'ms')
        return np.array([T_l, T_r])
        
    def LocalCurrent(self, e):
        I = np.zeros((2,2,self.N, self.N))
        for Spin in range(1+self.bSpin):
            #Use explicit expression in the case of coherent transport
            if self.Gam_b==0 or True:
                mu_L = e+self.Bias/2
                mu_R = e-self.Bias/2
                ei, ej = np.meshgrid(np.conjugate(self.eigval[Spin]), self.eigval[Spin])
                Ups = 1/(ei-ej)*(np.log((mu_R-ei)/(mu_L-ei))-np.log((mu_R-ej)/(mu_L-ej)))
                I[Spin] = -self.LocalTransmissionAnalytical([Ups, np.zeros((self.N, self.N))], Spin)
                
            #Integrate over energy for incoherent transport
            else:      
                E = LogLinLogSpace([np.min(self.e_arr)-100*(np.max(abs(self.gam))), np.min(self.e_arr)-10*(np.max(abs(self.gam))), e+self.Bias, e+self.Bias], N=1000)       
                Ups = np.zeros((2, self.N, self.N), dtype=complex)
                de = E[1:]-E[0:-1]
                for k, ek in enumerate((E[1:]+E[0:-1])/2):   
                    ei, ej = np.meshgrid(1/(ek-np.conjugate(self.eigval[Spin])), 1/(ek-self.eigval[Spin]))
                    Ups[0] += de[k]*ei*ej
                    Ups[1] += de[k]*ei*ej*(ek<=e)
                I[Spin] = self.LocalTransmissionAnalytical(Ups, Spin=Spin) 
                
        if not self.bSpin:
            I[1] = I[0]

        #Set to nanoAmpere units 
        I *= 1/(2*np.pi)*0.00662361795*10**9
        return I
        

        

if __name__ == '__main__':
    from Molecule import *
    from Graphics import Graphics

#    Mol = CGNR(23, 8, 8, 1, 4)
    Mol = ZGNR(5, 100)
    Mol.SetGam_b(0.05)
    Mol.UpdateMolecule(bNext=False)
    Mol.UpdateSystem(bNext=False)
    Mol.UpdateGates(Gmin=-3+Mol.Efermi, Gmax=3+Mol.Efermi)
    Mol.GetDOS()
    
    Graph = Graphics()
    Graph.SetMolecule(Mol)
    plt.show()

    
   


#    Mol1 = Molecule()
#    Mol1.AddAtom('C', 0, 0, 0)
#    Mol1.AddAtom('C', 1.41, 0, 0)
#    Mol1.SetLead(0, 'Left')
#    Mol1.SetLead(0, 'Right')
#    Mol1.SetGam_LR(5.6, 5.6)
#    Mol1 = Graphene(3,3, 'Armchair')
#    Mol1 = Graphene(6,401, 'Zigzag')
#    Mol1 = Graphene(4, 41, 'Zigzag')
#    Atom, x, y, z = Mol1.GetCoord()
#    xavg = np.mean(x)
#    for i in reversed(range(len(x))):
#        if (x[i]<(xavg-9*1.42) or x[i]>(xavg+9*1.42)) and y[i]<1.5:
#            Mol1.UpdateAtom(i, 'DEL')
#    Mol1.SetLeads()  
#    delta = 0.25   
#    Mol1.ShiftAtom(12, [-delta, delta, 0])
#    Mol1.ShiftAtom(69, [delta, -delta, 0])
#    Mol1.ShiftAtom(23, [delta, delta, 0])
##    Mol1.ShiftAtom(82, [-delta, -delta, 0])
#    Graph = Graphics()
#    Graph.SetMolecule(Mol1)
    
#    Mol1.ExportTransmission()
#    Mol1.ExportDOS()
    
  
    
#    TGreen = np.zeros((6, len(Mol1.Gates)))
#    TAnal  = np.zeros((6, len(Mol1.Gates)))
#    atoms = np.array([[371,372], [69,12], [174,70],[387,388],[291,187],[494,495]])
#    for i, ei in enumerate(Mol1.Gates):
#        TGr = Mol1.LocalTransmissionGreen(ei, [1,0])
#        TAn = Mol1.LocalTransmissionAnalytical(ei, [1,0])
#        for j, atom in enumerate(atoms):
#            TGreen[j, i] = TGr[atom[0], atom[1]]
#            TAnal [j, i] = TAn[atom[0], atom[1]]
#            print(i,ei)
#    fig = plt.figure()
#    for j, atom in enumerate(atoms):
#        ax = fig.add_subplot(2,3,j+1)
#        ax.semilogy(Mol1.Gates, abs(TGreen[j]), 'g')
#        ax.semilogy(Mol1.Gates, abs(TAnal[j]),  'g:')
#        ax.set_title(str(atom[0]) + 'to', str(atom[1]))
    
    
    
    
#    """Orthonormality eigenvectors"""
#    print('Difference real:', np.sum(abs(np.real(np.dot(Mol1.eigvec[0], np.transpose(Mol1.eigvec[0])))-I)))
#    print('Difference imag:', np.sum(abs(np.imag(np.dot(Mol1.eigvec[0], np.transpose(Mol1.eigvec[0])))-I)))
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    cax1 = ax1.matshow(np.real(np.dot(Mol1.eigvec[0], np.transpose(Mol1.eigvec[0]))))
#    cbax1 = fig.add_axes([0.05, 0.15, 0.4, 0.03]) 
#    plt.colorbar(cax1, orientation='horizontal', cax=cbax1)
#    cax1 = ax2.matshow(np.imag(np.dot(Mol1.eigvec[0], np.transpose(Mol1.eigvec[0]))))
#    cbax1 = fig.add_axes([0.55, 0.15, 0.4, 0.03])  
#    plt.colorbar(cax1, orientation='horizontal', cax=cbax1)

#    """Compare Greens functions total"""
#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,2,1)
#    ax2 = fig.add_subplot(2,2,2)
#    ax3 = fig.add_subplot(2,2,3)
#    ax4 = fig.add_subplot(2,2,4)
#    eps = np.zeros((2,len(Mol1.Gates)))    
#    Gr1  = np.zeros((2, len(Mol1.Gates)))
#    Gr2  = np.zeros((2, len(Mol1.Gates)))
#    for i, ei in enumerate(Mol1.Gates):
#        print(i, ei)
#        G1 = Mol1.GreenInv(ei)
#        G2 = Mol1.GreenDiag(ei)
#        eps[0,i] = np.sum(abs(np.real(G1-G2)))
#        eps[1,i] = np.sum(abs(np.imag(G1-G2)))
#        Gr1[0,i] = np.sum(np.real(G1))
#        Gr1[1,i] = np.sum(np.imag(G1))
#        Gr2[0,i] = np.sum(np.real(G2))
#        Gr2[1,i] = np.sum(np.imag(G2))
#    ax1.plot(Mol1.Gates, Gr1[0], 'b', lw=2, alpha=0.5)
#    ax2.plot(Mol1.Gates, Gr1[1], 'b', lw=2, alpha=0.5)
#    ax1.plot(Mol1.Gates, Gr2[0], 'g', ls=':')
#    ax2.plot(Mol1.Gates, Gr2[1], 'g', ls=':')
#    ax3.semilogy(Mol1.Gates, eps[0], 'r')
#    ax4.semilogy(Mol1.Gates, eps[1], 'r')
#    ax1.legend(['Green Inverse','Green Diagonalized'])
#    ax3.legend(['Summed absolute difference'])
#    ax1.set_title('Real part')
#    ax2.set_title('Imaginary part')
    

    
#    """Compare Greens functions at certain energy"""
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    e = -2.81
#    G1 = Mol1.GreenInv(e)
#    print(G1[0,2])
#    G2 = Mol1.GreenDiag(e)
#    print(G2[0,2])
#    cax = ax1.matshow(np.real(G1-G2))
#    print(np.real(G1-G2))
#    cax = ax2.matshow(np.imag(G1-G2))
#    ax1.set_title('Real part')
#    ax2.set_title('Imaginary part')
    
    
#    """Compare Greens functions at all energies as a matrix"""
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    G1 = np.zeros((Mol1.N, Mol1.N))
#    G2 = np.zeros((Mol1.N, Mol1.N))
#    for i, ei in enumerate(Mol1.Gates):
#        Gr1 = Mol1.GreenInv(ei)
#        Gr2 = Mol1.GreenDiag(ei)
#        G1 += abs(np.real(Gr1-Gr2))
#        G2 += abs(np.imag(Gr1-Gr2))
#    cax1 = ax1.imshow(G1, interpolation='nearest')
#    cax2 = ax2.matshow(G2)
#    cbax1 = fig.add_axes([0.05, 0.15, 0.4, 0.03]) 
#    plt.colorbar(cax1, orientation='horizontal', cax=cbax1)
#    cbax2 = fig.add_axes([0.55, 0.15, 0.4, 0.03]) 
#    plt.colorbar(cax2, orientation='horizontal', cax=cbax2)
#    ax1.set_title('Real part')
#    ax2.set_title('Imaginary part')
    
#    """Take out one of the points"""
#    fig = plt.figure()
#    ax1 = fig.add_subplot(2,2,1)
#    ax2 = fig.add_subplot(2,2,2)
#    ax3 = fig.add_subplot(2,2,3)
#    ax4 = fig.add_subplot(2,2,4)
#    alpha = 0
#    beta  = 1
#    Gr1  = np.zeros((3, len(Mol1.Gates)))
#    Gr2  = np.zeros((3, len(Mol1.Gates)))
#    num  = np.zeros((2, len(Mol1.Gates), Mol1.N))
#    den  = np.zeros((2, len(Mol1.Gates), Mol1.N))
#    NUM = (Mol1.eigvec[0][alpha, :]*Mol1.eigvec[0][beta, :])
#    for i, ei in enumerate(Mol1.Gates):
#        G1 = Mol1.GreenInv(ei)[alpha,beta]
#        G2 = Mol1.GreenDiag(ei)[alpha, beta]
#        Gr1[0,i] = np.sum(np.real(G1))
#        Gr1[1,i] = np.sum(np.imag(G1))
#        Gr2[0,i] = np.sum(np.real(G2))
#        Gr2[1,i] = np.sum(np.imag(G2))
#        DEN = ei-Mol1.eigval[0]
#        num[0,i] = (np.real(NUM)*np.real(DEN)+np.imag(NUM)*np.imag(DEN))
#        num[1,i] = -(np.imag(NUM)*np.real(DEN)+np.real(NUM)*np.imag(DEN))
#        den[0,i] = (np.real(DEN)**2+np.imag(DEN)**2)
#        den[1,i] = den[0,i]
#    ax1.plot(Mol1.Gates, Gr1[0], 'b', lw=2, alpha=0.5)
#    ax2.plot(Mol1.Gates, Gr1[1], 'b', lw=2, alpha=0.5)
#    ax1.plot(Mol1.Gates, Gr2[0], 'g', ls=':')
#    ax2.plot(Mol1.Gates, Gr2[1], 'g', ls=':')
##    ax1.plot(Mol1.Gates, abs(Gr1[0]-Gr2[0]), 'r', ls=':')
##    ax2.plot(Mol1.Gates, abs(Gr1[1]-Gr2[1]), 'r', ls=':')
##    ax1.plot(Mol1.Gates, np.sum(num[0]/den[0], axis=1), 'k')
##    ax2.plot(Mol1.Gates, np.sum(num[1]/den[1], axis=1), 'k')
#    ax1.set_title('Real part')
#    ax2.set_title('Imaginary part')
##    ax1.legend(['Green Inverted', 'Green Diagonalized'])
#    for n in range(Mol1.N):
##        if n is not 4:
##            continue
##        print('Energy is', Mol1.eigval[0][4])
##        print('Eigenstate is', Mol1.eigvec[0][4])
#        c = np.random.rand(3)
#        ax3.plot(Mol1.Gates, num[0,:,n], c=c, lw=1, alpha=0.5)
#        ax4.plot(Mol1.Gates, num[1,:,n], c=c, lw=1, alpha=0.5)
#        ax3.plot(Mol1.Gates, den[0,:,n], c=c, ls=':')
#        ax4.plot(Mol1.Gates, den[1,:,n], c=c, ls=':')
#        ax3.plot(Mol1.Gates, num[0,:,n]/den[0,:,n], c=c, lw=2)
#        ax4.plot(Mol1.Gates, num[1,:,n]/den[1,:,n], c=c, lw=2)
#    ax3.legend(['Numerator', 'Denominator', 'Sum'])
#    print('Eigenvalues \n', Mol1.eigval[0])
#    
#    """Test DOS"""
#    ax1 = fig.add_subplot(1,1,1)
#    DOS1 = np.sum(Mol1.CalcDOSGreen(), axis=0)
#    DOS2 = np.sum(Mol1.CalcDOSAnalytical(), axis=0)
#    LDOS3 = np.zeros(len(Mol1.Gates))
#    for i, ei in enumerate(Mol1.Gates):
#        LDOS3[i] = np.sum(Mol1.LDOSAnalytical(ei))
#    ax1.plot(Mol1.Gates, DOS1, 'pink', lw=2, ls=':')
#    ax1.plot(Mol1.Gates, DOS2, 'seagreen', lw=2, alpha=0.4)
#    ax1.plot(Mol1.Gates, LDOS3, 'k', lw=4, alpha=0.3)
#    ax1.legend(['Inverted Green', 'Diag Green', 'Summed LDOS'])
    
#    """Test LDOS for random atoms"""
#    LDOS1 = np.zeros((Mol1.N, len(Mol1.Gates)))
#    LDOS2 = np.zeros((Mol1.N, len(Mol1.Gates)))
#    LDOS3 = np.zeros((Mol1.N, len(Mol1.Gates)))
#    for i, ei in enumerate(Mol1.Gates):
#        LDOS1[:,i] = np.sum(Mol1.LDOSGreen(ei, bTemp=True), axis=0)
#        LDOS2[:,i] = np.sum(Mol1.LDOSGreen(ei, bTemp=False), axis=0)
#        LDOS3[:,i] = np.sum(Mol1.LDOSAnalytical(ei), axis=0)
#
#    C=6
#    for i, n in enumerate(np.random.randint(0, Mol1.N-1, C**2)):
#        print(n)
#        ax1 = fig.add_subplot(C, C, i+1)
#        ax1.plot(Mol1.Gates, LDOS3[n], 'k', ls=':', lw='3')
#        ax1.plot(Mol1.Gates, LDOS2[n], 'r', lw=2, alpha=0.6)
##        ax1.plot(Mol1.Gates, LDOS3[n], 'g', alpha=0.4)
#        ax1.set_title(str(n))
#    ax1.legend(['Inverted Green', 'Diag Green', 'Explicit'])
    
#    """"""
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    Shalf = scLA.sqrtm(Mol1.S)
#    I = np.dot(np.transpose(Mol1.eigvec[0]), Shalf)
#    I = np.dot(Mol1.eigvec[0], I)
#    I = np.dot(Shalf, I)
#    I = np.dot(Mol1.eigvec[0], np.transpose(Mol1.eigvec[0]))
##    I = la.inv(Mol1.eigvec[0])-np.transpose(Mol1.eigvec[0])
#    print(I)
#    print(np.sum((I-np.identity(Mol1.N))**2))
#    ax1.matshow(np.real(I))
#    ax2.matshow(Shalf)
#    ax1.plot(np.real(I.diagonal()), 'k')
#    ax2.plot(np.imag(I.diagonal()), 'k', ls=':')
#    ax1.plot(np.real(Mol1.eigval[0]), 'r')
#    ax2.plot(np.imag(Mol1.eigval[0]), 'r', ls=':')
#    plt.show()



