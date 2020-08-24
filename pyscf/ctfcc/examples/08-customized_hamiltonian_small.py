#!/usr/bin/env python

'''
For small Hamiltonian, one can afford to store one copy of integrals on each process
Six-site 1D U/t=2 Hubbard-like model system with PBC at half filling.
The model is gapped at the mean-field level
'''

import numpy
from pyscf import gto, scf, ao2mo
from pyscf.ctfcc import rccsd
import ctf
asarray = ctf.astensor
einsum = ctf.einsum

U = 2.0
n = 6

h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = U
# h1 and eri stored on each process

mol = gto.M(verbose=4)
mol.nelectron = n
mol.incore_anyway = True

# mf ran on each process
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)
mf._eri = ao2mo.restore(8, eri, n)
mf.kernel()

# rewrite the routine for ao2mo in ctfcc.rccsd
def make_ao_ints(mol, mo_coeff, nocc):
    '''
    ao2mo for pyscf.ctfcc.rccsd is based on the ctfcc.integrals.ao2mo.make_ao_ints functions,
    specifically this func should return partially transformed chemists eri

    return a list of integrals stored as ctf tensors:
        ppoo = (uv|ij) = (uv|rs) C*_{ri} C_{sj}
        ppov = (uv|ia) = (uv|rs) C*_{ri} C_{sa}
        ppvv = (uv|ab) = (uv|rs) C*_{ra} C_{sb}
    '''
    int4c = asarray(eri) # convert numpy array to distributed ctf array
    mo = asarray(mo_coeff)
    ppoo = ctf.einsum('uvrs,ri,sj->uvij', int4c, mo[:,:nocc].conj(), mo[:,:nocc])
    ppov = ctf.einsum('uvrs,ri,sa->uvia', int4c, mo[:,:nocc].conj(), mo[:,nocc:])
    ppvv = ctf.einsum('uvrs,ra,sb->uvab', int4c, mo[:,nocc:].conj(), mo[:,nocc:])
                        # use ctf.einsum for ao2mo transformation
    return ppoo, ppov, ppvv

rccsd.make_ao_ints = make_ao_ints
mycc = rccsd.RCCSD(mf)
mycc.kernel()
