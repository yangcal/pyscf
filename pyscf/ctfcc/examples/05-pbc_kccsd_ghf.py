#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

from pyscf.pbc import scf, gto, cc
from pyscf.ctfcc import kccsd
from pyscf.ctfcc import ctf_helper

rank = ctf_helper.rank

cell = gto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
H 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 4
cell.spin = 3
cell.build()

kpts = cell.make_kpts([1,1,3])
mf = scf.KUHF(cell, kpts, exxdiv=None)

if rank==0: #SCF only needs to run on one process
    mf.kernel()

# broadcast the scf results to all mf obj
ctf_helper.synchronize(mf, ["mo_coeff", "mo_occ", "mo_energy", "e_tot"])
mf = mf.to_ghf(mf)
###################################################
###############  Ground State CCSD  ###############
###################################################
mycc = kccsd.KGCCSD(mf)
mycc.conv_tol = 1e-10
eris = mycc.ao2mo()
ecc = mycc.kernel(eris=eris)[0]
print("Eccsd = ", ecc--0.008152029750336229)

###################################################
####################  CCSD(T)  ####################
###################################################
et = mycc.ccsd_t(eris=eris, slice_size=4000)
print("Eccsd(t) = ", et--9.166178879353904e-06)

###################################################
####################  EOM-IP  #####################
###################################################
eip, vip = mycc.ipccsd(nroots=2, kptlist=[1])
print(eip[0,0]--0.11536587588326533, eip[0,1]-0.3571276915356543)

###################################################
####################  EOM-EA  #####################
###################################################
eea, vea = mycc.eaccsd(nroots=2, kptlist=[2])
print(eea[0,0]-0.5584017179972866, eea[0,1]-1.2919236453028176)
