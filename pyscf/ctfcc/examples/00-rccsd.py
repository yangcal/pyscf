#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
molecular RCCSD with ctf parallelization
Usage: mpirun -np 4 python 00-rccsd.py
'''
from pyscf import scf, gto
from pyscf.ctfcc import rccsd
from pyscf.ctfcc.ctf_helper import rank

mol = gto.Mole(atom = 'H 0 0 0; F 0 0 1.1',
               basis = 'ccpvdz',
               verbose= 4)
mol.build()
mf = scf.RHF(mol)

if rank==0: #SCF only needs to run on one process
    mf.kernel()
###################################################
###############  Ground State CCSD  ###############
###################################################
mycc = rccsd.RCCSD(mf)
eris = mycc.ao2mo()
ecc = mycc.kernel(eris=eris)[0]
print("Eccsd Err =", ecc--0.21633762905952342)

###################################################
####################  CCSD(T)  ####################
###################################################
et = mycc.ccsd_t(eris=eris, slice_size=4000) # use 4GB for slicing the t3 amplitudes
print("Eccsd(t) Err =", et--0.0024136911479467732)

###################################################
####################  EOM-IP  #####################
###################################################
eip, vipr = mycc.ipccsd(nroots=2, koopmans=True)
eip1, vipl = mycc.ipccsd(nroots=2, koopmans=True, left=True)
print("EOM-IP Err")
print(eip[0]-0.5325135626202897, eip[1]-0.5325135626202897)
print(eip1[0]-0.5325135692606366, eip1[1]-0.5325135692606366)

###################################################
##################  EOM-IP-star  ##################
###################################################
eip_star = mycc.ipccsd_star_contract(eip, vipr, vipl)
print("EOM-IP-star Err")
print(eip_star[0]-0.5405005695953484, eip_star[1]-0.5405005695953494)

###################################################
####################  EOM-EA  #####################
###################################################
eea, vear = mycc.eaccsd(nroots=2, koopmans=True)
eea1, veal = mycc.eaccsd(nroots=2, koopmans=True, left=True)
print("EOM-EA Err")
print(eea[0]-0.12686682891578827, eea[1]-0.6381980171319832)
print(eea1[0]-0.1268668128198946, eea1[1]-0.6381980040077034)

###################################################
##################  EOM-EA-star  ##################
###################################################
eea_star = mycc.eaccsd_star_contract(eea, vear, veal)
print("EOM-EA-star Err")
print(eea_star[0]-0.12573837511716066, eea_star[1]-0.6343997731168681)


###################################################
###################  EOM-IP-T*  ###################
###################################################
eip_t = mycc.ipccsd_t_star(nroots=2)[0]
print("EOM-IP_Ta Err")
print(eip_t[0]-0.53270091671908, eip_t[1]-0.53270091671908)

###################################################
###################  EOM-EA-T*  ###################
###################################################
eea_t = mycc.eaccsd_t_star(nroots=2)[0]
print("EOM-EA_Ta Err")
print(eea_t[0]-0.1279616622425941, eea_t[1]-0.5575046127343419)
