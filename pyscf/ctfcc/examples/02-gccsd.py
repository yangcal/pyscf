#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
molecular GCCSD with ctf parallelization
Usage: mpirun -np 4 python 02-gccsd.py
'''
from pyscf import scf, gto, cc
from pyscf.ctfcc import gccsd
from pyscf.ctfcc.ctf_helper import rank, synchronize

mol = gto.Mole()
mol.atom = [['O', (0.,   0., 0.)],
            ['O', (1.21, 0., 0.)]]
mol.basis = 'sto3g'
mol.charge = 1
mol.spin = 1
mol.verbose = 0
mol.build()


mf = scf.UHF(mol)
if rank==0: #SCF only needs to run on one process
    mf.kernel()
synchronize(mf, ["mf_coeff", "mf_occ", "mf_energy", "e_tot"])
mf = mf.to_ghf()

###################################################
###############  Ground State CCSD  ###############
###################################################
mycc = gccsd.GCCSD(mf)

eris = mycc.ao2mo()
ecc = mycc.kernel(eris=eris)[0]
print("Eccsd Err =", ecc--0.19680960277044784)

###################################################
####################  CCSD(T)  ####################
###################################################
et = mycc.ccsd_t(eris=eris, slice_size=4000) # use 4GB for slicing the t3 amplitudes
print("Eccsd(t) Err =", et--0.0020704574463285435)

###################################################
####################  EOM-IP  #####################
###################################################
#from pyscf.cc import eom_gccsd
#eom = eom_gccsd.EOMIP(mycc)
eip, vipr = mycc.ipccsd(nroots=2)
eip1, vipl = mycc.ipccsd(nroots=2, left=True)
print("EOM-IP Err")
print(eip[0]-0.8770943434864885, eip[1]-1.0792530341475137)
print(eip1[0]-0.8770943519482644, eip1[1]-1.0792530386166537)

###################################################
##################  EOM-IP-star  ##################
###################################################

eip_star = mycc.ipccsd_star_contract(eip, vipr, vipl)
print("EOM-IP-star Err")
print(eip_star[0]-0.8912704947395196, eip_star[1]-1.044712626228977)


###################################################
####################  EOM-EA  #####################
###################################################
#eom = eom_gccsd.EOMEA(mycc)
eea, vear = mycc.eaccsd(nroots=2)
eea1, veal = mycc.eaccsd(nroots=2, left=True)
print("EOM-EA Err")
print(eea[0]--0.3006404309697215, eea[1]--0.290227350388533)
print(eea1[0]--0.30064043097137855, eea1[1]--0.29022735038427083)

###################################################
##################  EOM-EA-star  ##################
###################################################

eea_star = mycc.eaccsd_star_contract(eea, vear, veal)
print("EOM-EA-star Err")
print(eea_star[0]--0.32413935579630115, eea_star[1]--0.31439872902230837)

###################################################
###################  EOM-IP-T*  ###################
###################################################
#eom = eom_gccsd.EOMIP_Ta(mycc)
eip_t = mycc.ipccsd_t_star(nroots=2)[0]
print("EOM-IP_Ta Err")
print(eip_t[0]-0.8777514380784694, eip_t[1]-1.0805396845045254)

###################################################
###################  EOM-EA-T*  ###################
###################################################
#eom = eom_gccsd.EOMEA_Ta(mycc)
eea_t = mycc.eaccsd_t_star(nroots=2)[0]
#eea_t = mycc.eaccsd_t_star(nroots=2)[0]
print("EOM-EA_Ta Err")
print(eea_t[0]--0.2995191771402712, eea_t[1]--0.2895448643879008)
