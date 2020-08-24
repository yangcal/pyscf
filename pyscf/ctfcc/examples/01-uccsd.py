#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
UCCSD with ctf parallelization
Usage: mpirun -np 4 python 01-uccsd.py
'''
from pyscf import scf, gto
from pyscf.ctfcc import uccsd
from pyscf.ctfcc.ctf_helper import rank

mol = gto.Mole()
mol.atom = [['O', (0.,   0., 0.)],
            ['O', (1.21, 0., 0.)]]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()

mf = scf.UHF(mol)
if rank==0: #SCF only needs to run on one process
    mf.kernel()
###################################################
###############  Ground State CCSD  ###############
###################################################
mycc = uccsd.UCCSD(mf)
eris = mycc.ao2mo()
ecc = mycc.kernel(eris=eris)[0]
print("Eccsd Err =", ecc--0.3522309099775712)

###################################################
####################  CCSD(T)  ####################
###################################################
et = mycc.ccsd_t(eris=eris, slice_size=4000) # use 4GB for slicing the t3 amplitudes
print("Eccsd(t) Err =", et--0.009831049952385255)

###################################################
####################  EOM-IP  #####################
###################################################
eip, vipr = mycc.ipccsd(nroots=2, koopmans=True)
print("EOM-IP Err")
print(eip[0]-0.4326031867701253, eip[1]-0.4326031867701253)

###################################################
####################  EOM-EA  #####################
###################################################
eea, vear = mycc.eaccsd(nroots=2, koopmans=True)
print("EOM-EA Err")
print(eea[0]-0.06490407831227946, eea[1]-0.06490407831227946)
