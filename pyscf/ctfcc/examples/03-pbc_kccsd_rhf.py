#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
A simple example to run pbc KRCCSD with ctf parallelization
Usage: mpirun -np 4 python 03-pbc_kccsd_rhf.py
'''
from pyscf.pbc import scf, gto
from pyscf.ctfcc import kccsd_rhf
from pyscf.ctfcc import ctf_helper

rank = ctf_helper.rank

cell = gto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KRHF(cell, kpts, exxdiv=None)

if rank==0: #SCF only needs to run on one process
    mf.kernel()

###################################################
###############  Ground State CCSD  ###############
###################################################
mycc = kccsd_rhf.KRCCSD(mf)
eris = mycc.ao2mo()
ecc = mycc.kernel()[0]
print("Eccsd = ", ecc--0.010315878072488838)

###################################################
####################  CCSD(T)  ####################
###################################################
et = mycc.ccsd_t(eris=eris, slice_size=4000) # use 4GB for slicing the t3 amplitudes
print("Eccsd(t) = ", et--5.1848049827171756e-06)

###################################################
####################  EOM-IP  #####################
###################################################
eip, vipr = mycc.ipccsd(nroots=2, kptlist=[0])
eip1, vipl = mycc.ipccsd(nroots=2, kptlist=[0], left=True)
print("IP")
print(eip[0,0]-0.052828726372268556, eip[0,1]-0.5814553488891845)
print(eip1[0,0]-0.05282873991815146, eip1[0,1]-0.5814552557174137)
###################################################
##################  EOM-IP-star  ##################
###################################################
eip_star = mycc.ipccsd_star_contract(eip[0], vipr[0], vipl[0], kshift=0)
print("IP*")
print(eip_star[0]-0.05220776396260876, eip_star[1]-0.5812773169106573)

###################################################
####################  EOM-EA  #####################
###################################################
eea, vear = mycc.eaccsd(nroots=2, kptlist=[1])
eea1, veal = mycc.eaccsd(nroots=2, kptlist=[1], left=True)
print("EA")
print(eea[0,0]-1.6093835115491955, eea[0,1]-2.2284005426098474)
print(eea1[0,0]-1.609383462775453, eea1[0,1]-2.2284005814533296)

###################################################
##################  EOM-EA-star  ##################
###################################################
eea_star = mycc.eaccsd_star_contract(eea[0], vear[0], veal[0], kshift=1)
print("EA*")
print(eea_star[0]-1.6093934772695793,eea_star[1]-2.2281606204020425)

###################################################
###################  EOM-IP-T*  ###################
###################################################
eip, vipr = mycc.ipccsd_t_star(nroots=2, kptlist=[0])
print("IP T*")
print(eip[0,0]-0.052842985493215575, eip[0,1]-0.581450758927699)

###################################################
###################  EOM-EA-T*  ###################
###################################################
eea, vear = mycc.eaccsd_t_star(nroots=2, kptlist=[1])
print("EA T*")
print(eea[0,0]-1.6093889022554788, eea[0,1]-2.2284009430958993)
