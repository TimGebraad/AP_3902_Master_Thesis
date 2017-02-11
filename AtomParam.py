""" Stores the relevant parameters for the used atoms """

"""Numbering of the atom type"""
AtomNr = { 1: 'H',
           5: 'B',
           6: 'C',
           7: 'N',
           8: 'O',
          16: 'S'}

""" Self energies of the atoms """
E0 = -5.0
e0 = {'e0':   E0,
      'H':    E0,
      'C':    E0-0.0,
      'B':    E0-1.7,           #verified
      'N':    E0-1.9,           #verified
      'O':    E0+2.6,           #verified
      'S':    E0-2.2}  
       
""" Neirest neighbour interaction between atoms """
tm = {'CC':-2.8, 'CH':  0.0, 'HC':  0.0,
                 'CN': -2.8, 'NC': -2.8,    #verified
                 'CB': -2.8, 'BC': -2.8,    #verified
                 'CO': -2.8, 'OC': -2.8,    #verifed
                 'CS':  6.6, 'SC':  6.6}    # To be verified
tm2 = {'CC':-0.27}

""" Mutual distance """
dL = 1.42 #Angstrom

""" Colors for displaying the atoms """
cAtom = {'H': 'lightblue',
         'C': 'k',
         'B': 'gray',
         'N': 'g',
         'O': 'b',
         'S': 'y'}

""" Initial interaction values and broadening factor"""
Gam_L  = 0.05 #0.316            #Left lead
Gam_R  = 0.05 #0.316           #Right lead
Gam_S  = (-0.4+0.1j)/1j   #Surface lead
Gam_b  = 0.05# 4*8.6*10**(-5)           #Broadening factor

""" Other parameters """
Overlap = 0.15
Hubbard = 8
