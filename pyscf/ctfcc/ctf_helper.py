import numpy
from pyscf import lib
import ctf
import sys, os

'''helper functions for ctf'''
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except:
    comm, rank, size = None, 0, 1

if rank!=0:
    sys.stdout = open(os.devnull, 'w')    # supress printing

def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    if comm is None:
        ntasks = len(tasks)
    else:
        ntasks = max(comm.allgather(stop-start))

    return tasks[start:stop], ntasks

def synchronize(obj, keys):
    if comm is not None:
        comm.barrier()
        for i in keys:
            val = comm.bcast(getattr(obj, i, None), root=0)
            setattr(obj, i, val)

def pack_tril(mat):
    if mat.ndim!=2:
        raise ValueError("Only 2D Matrix supported")
    nd = mat.shape[0]
    vec_size = nd*(nd+1)//2
    ind, val = mat.read_local()
    out = ctf.zeros([vec_size], dtype=mat.dtype)
    idxi, idxj = numpy.unravel_index(ind, mat.shape)
    mask = idxi >= idxj
    ind = idxi * (idxi+1)//2 + idxj
    ind = ind[mask]
    val = val[mask]
    out.write(ind, val)
    return out


def unpack_tril(tril, filltriu=lib.HERMITIAN, axis=-1, out=None):
    if tril.ndim != 1:
        raise ValueError("unpack_tril only support 1D vector")

    count, nd = 1, tril.size
    nd = int(numpy.sqrt(nd*2))
    shape = (nd,nd)

    if out is None:
        out = ctf.zeros(shape, dtype=tril.dtype)
    ind, val  =  tril.read_local()
    idxi  = numpy.ceil(numpy.sqrt(ind*2+2.25)-1.5)
    idxi = numpy.asarray(idxi, dtype=int)
    idxj = ind -  idxi * (idxi+1)//2
    idxij = idxi * nd + idxj
    if filltriu == lib.PLAIN:
        ind = idxij
    elif filltriu == lib.HERMITIAN:
        mask = idxi != idxj
        idxji = idxj * nd + idxi
        ind = numpy.concatenate((idxij, idxji[mask]))
        val = numpy.concatenate((val, val[mask].conj()))
    elif filltriu == lib.ANTIHERMI:
        idxji = idxj * nd + idxi
        ind = numpy.concatenate((idxij, idxji))
        val = numpy.concatenate((val, -val.conj()))
    elif filltriu == lib.SYMMETRIC:
        mask = idxi != idxj
        idxji = idxj * nd + idxi
        ind = numpy.concatenate((idxij, idxji[mask]))
        val = numpy.concatenate((val, val[mask]))
    out.write(ind, val)
    return out

def amplitudes_to_vector_s4(t1, t2):
    nocc, nvir = t1.shape
    off = nvir*(nvir-1)//2

    ind, val = t2.read_local()
    iind, jind, aind, bind = numpy.unravel_index(ind, t2.shape)
    mask = numpy.logical_and(iind>jind, aind>bind)
    idxij = iind * (iind-1)//2 + jind
    idxab = aind * (aind-1)//2 + bind
    ind = idxij[mask] * off + idxab[mask]
    t2vec = ctf.zeros([nocc*(nocc-1)*nvir*(nvir-1)//4], dtype=t2.dtype)
    t2vec.write(ind, val[mask])
    return ctf.hstack((t1.ravel(), t2vec))

def vector_to_amplitudes_s4(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    t1 = vector[:nov].reshape(nocc,nvir)
    t2vec = vector[nov:].reshape(nocc*(nocc-1)//2, nvir*(nvir-1)//2)
    ind, val = t2vec.read_local()
    indij, indab = numpy.unravel_index(ind, t2vec.shape)

    idxi = numpy.floor(numpy.sqrt(indij*2+.25)+0.5)
    idxi = numpy.asarray(idxi, dtype=int)
    idxj = indij -  idxi * (idxi-1)//2
    idxa = numpy.floor(numpy.sqrt(indab*2+.25)+0.5)
    idxa = numpy.asarray(idxa, dtype=int)
    idxb = indab -  idxa * (idxa-1)//2

    shape = (nocc,nocc,nvir,nvir)
    t2 = ctf.zeros([nocc,nocc,nvir,nvir], dtype=t2vec.dtype)
    ind = numpy.hstack((numpy.ravel_multi_index((idxi,idxj,idxa,idxb), shape), \
                        numpy.ravel_multi_index((idxj,idxi,idxa,idxb), shape), \
                        numpy.ravel_multi_index((idxi,idxj,idxb,idxa), shape), \
                        numpy.ravel_multi_index((idxj,idxi,idxb,idxa), shape)))
    val = numpy.hstack((val, -val, -val, val))
    t2.write(ind, val)
    return t1, t2

def pack_ip_r2(r2):
    nocc, nvir = r2.shape[1:]
    r2vec = ctf.zeros([nocc*(nocc-1)//2*nvir], dtype=r2.dtype)
    ind, val = r2.read_local()
    idxi, idxj, idxa = numpy.unravel_index(ind, r2.shape)
    mask = idxi > idxj
    idxij = idxi*(idxi-1)//2+idxj
    idxija = idxij[mask]*nvir + idxa[mask]
    r2vec.write(idxija, val[mask])
    return r2vec

def unpack_ip_r2(r2vec, nmo, nocc):
    nvir = nmo - nocc
    ind, val = r2vec.read_local()
    indij, idxa = numpy.unravel_index(ind, (nocc*(nocc-1)//2, nvir))
    idxi = numpy.floor(numpy.sqrt(indij*2+.25)+0.5)
    idxi = numpy.asarray(idxi, dtype=int)
    idxj = indij - idxi * (idxi-1)//2
    r2aaa = ctf.zeros([nocc,nocc,nvir], dtype=r2vec.dtype)
    ind = numpy.hstack((numpy.ravel_multi_index((idxi,idxj,idxa), r2aaa.shape), \
                        numpy.ravel_multi_index((idxj,idxi,idxa), r2aaa.shape)))
    val = numpy.hstack((val, -val))
    r2aaa.write(ind, val)
    return r2aaa

def pack_ea_r2(r2):
    nocc, nvir = r2.shape[:2]
    r2v = ctf.zeros([nocc*nvir*(nvir-1)//2], dtype=r2.dtype)
    ind, val = r2.read_local()
    idxi, idxa, idxb = numpy.unravel_index(ind, r2.shape)
    mask = idxa > idxb
    idxab = idxa*(idxa-1)//2+idxb
    ind = idxab[mask] + idxi[mask]*nvir*(nvir-1)//2
    r2v.write(ind, val[mask])
    return r2v

def unpack_ea_r2(r2v, nmo, nocc):
    nvir = nmo - nocc
    r2aaa = ctf.zeros([nocc,nvir,nvir], dtype=r2v.dtype)
    ind, val = r2v.read_local()
    idxi, idxab = numpy.unravel_index(ind, (nocc,nvir*(nvir-1)//2))
    idxa = numpy.floor(numpy.sqrt(idxab*2+.25)+0.5)
    idxa = numpy.asarray(idxa, dtype=int)
    idxb = idxab - idxa * (idxa-1)//2

    ind = numpy.hstack((numpy.ravel_multi_index((idxi,idxa,idxb), r2aaa.shape), \
                        numpy.ravel_multi_index((idxi,idxb,idxa), r2aaa.shape)))
    val = numpy.hstack((val, -val))
    r2aaa.write(ind, val)
    return r2aaa
