#!/usr/bin/env python

'''
I. For large systems, entire 4-centered integrals stored/computed on one process.
II. SCF only performed on one process then synchronized
III. Two different methods provided to write integrals as ctf arrays
     see Method A, B below
'''

import numpy
from pyscf import gto, scf, ao2mo
from pyscf.ctfcc import rccsd
from pyscf.ctfcc import ctf_helper

rank = ctf_helper.rank
comm = ctf_helper.comm

import ctf
asarray = ctf.astensor
einsum = ctf.einsum

U = 4.0
n = 60 # note: this is an unconverged system(60 site Hubbard)
       # meant just for demonstration

mol = gto.M(verbose=4)
mol.nelectron = n
mol.incore_anyway = True

h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = -1.0

mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: numpy.eye(n)

'''
Method A: eri stored and written as ctf array with just on process
'''
# A1: initialze an empty distributed ctf tensor
int4c = ctf.zeros([n,n,n,n])

if rank==0:
    eri = numpy.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = U # eri tensor is only initialzed on current process
    mf._eri = ao2mo.restore(8, eri, n)
    mf.kernel() # scf performed on rank 0
    dm = mf.make_rdm1()
    fock = mf.get_veff(mol, dm)
    # A2: converts numpy array to ctf array on this process
    int4c.write(numpy.arange(eri.size), eri.ravel())
else:
    fock = None
    int4c.write([], []) # A2: other process writes empty block, (required by CTF)

fock = comm.bcast(fock, root=0)
mf.get_veff = lambda *args: fock # overwrite get_veff to return a constant
                                 # so eri is not needed on each process

'''
Method B: nonzero blocks of the eri stored and written to ctf tensor on all processes
'''

def write_eri_parallel(n, U):
    # B1: initialze an empty ctf tensor
    integrals = ctf.zeros([n,n,n,n])
    # B2: assign different tasks to each process
    tasks, ntasks = ctf_helper.static_partition(numpy.arange(n))
    # tasks: a list of sites assigned to current process
    # ntasks: maximal number of sites on any process
    for i in range(ntasks):
        if i >= len(tasks):
             # B3: empty write needed for ctf synchronization
            integrals.write([], [])
        else:
            # B3: compute 1d flattened index and write
            ind =  tasks[i]*(n**3+n**2+n+1)
            integrals.write([ind], [U])
    return integrals

int4c = write_eri_parallel(n, U)

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
    mo = asarray(mo_coeff)
    ppoo = ctf.einsum('uvrs,ri,sj->uvij', int4c, mo[:,:nocc].conj(), mo[:,:nocc])
    ppov = ctf.einsum('uvrs,ri,sa->uvia', int4c, mo[:,:nocc].conj(), mo[:,nocc:])
    ppvv = ctf.einsum('uvrs,ra,sb->uvab', int4c, mo[:,nocc:].conj(), mo[:,nocc:])
    return ppoo, ppov, ppvv

rccsd.make_ao_ints = make_ao_ints

mycc = rccsd.RCCSD(mf)
mycc.kernel()
