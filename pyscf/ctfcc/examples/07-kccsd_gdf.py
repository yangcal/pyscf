#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Kpoint KCCSD with GDF integrals
Usage: mpirun -np 4 python 07-kccsd_ghf.py
'''
from mpi4py import MPI
from pyscf.pbc import scf, gto, cc
from pyscf.ctfcc.kccsd_rhf import KRCCSD
from pyscf.ctfcc.integrals import mpigdf
from pyscf.ctfcc.ctf_helper import rank, comm

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.mesh = [15,15,15]
cell.build()

kpts = cell.make_kpts([1,1,3])

###################################################
###########  GDF from serial pyscf run  ###########
###################################################

mf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
if rank==0:
    ehf = mf.kernel()
    print("Ehf Err:", ehf - -9.460148501955697)
# CC will convert the GDF eri on disk to ctf tensors (with one process)
mycc = KRCCSD(mf)
ecc = mycc.kernel()[0]
print("Ecc Err:", ecc--0.16522692363262184)

###################################################
################  GDF from mpigdf  ################
###################################################
mf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
mydf = mpigdf.GDF(cell, kpts)
mydf.build()
mydf.dump_to_file() # dump the eri to disk so scf can run on top of it
                    # can specify filename with mydf.dump_to_file(cderi_file='gdf.eri')
mf.with_df = mydf
if rank==0:
    ehf = mf.kernel()
    print("Ehf Err:", ehf - -9.460148501955697)

mycc = KRCCSD(mf)
ecc = mycc.kernel()[0]
print("Ecc Err:", ecc--0.16522692363262184)
