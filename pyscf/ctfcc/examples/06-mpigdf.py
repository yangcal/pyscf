#!/usr/bin/env python
#
# Author: Yang Gao <younggao1994@gmail.com>
#

'''
Using MPIGDF to generate ERIs for pyscf.pbc
Usage: mpirun -np 4 python 06-mpigdf.py
'''
from mpi4py import MPI
from pyscf.pbc import gto, scf, df
from pyscf.ctfcc.integrals import mpigdf
from pyscf.ctfcc import ctf_helper

rank = ctf_helper.rank

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
cell.build()

kpts = cell.make_kpts([1,1,3])

mydf= mpigdf.GDF(cell, kpts)
mydf.build()
mydf.dump_to_file()
if rank==0:
    mf= scf.KRHF(cell,kpts)
    mf.with_df = mydf
    e1 =mf.kernel()
    print(e1--10.510484874469004)
