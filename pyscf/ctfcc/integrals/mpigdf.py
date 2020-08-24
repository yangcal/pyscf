#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yang Gao <younggao1994@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>

import time
import numpy
import ctf
import scipy
from pyscf.pbc import df
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import ft_ao
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member, unique,
                                       KPT_DIFF_TOL)
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv
from pyscf.ctfcc import ctf_helper

"""
Parallel GDF. GDF integrals stored in memory as distributed ctf tensors
This module is ctf-specific
"""
comm = ctf_helper.comm
rank = ctf_helper.rank

def get_member(kpti, kptj, kptij_lst):
    kptij = numpy.vstack((kpti,kptj))
    ijid = member(kptij, kptij_lst)
    dagger = False
    if len(ijid) == 0:
        kptji = numpy.vstack((kptj,kpti))
        ijid = member(kptji, kptij_lst)
        dagger = True
    return ijid, dagger

def from_serial(mydf):
    """
    convert a serial pyscf.pbc.df.GDF object to cc_sym.mpigdf.GDF object when integrals are already generated
    Note: only one process is used for reading from disk
    """
    newdf = GDF(mydf.cell, mydf.kpts)
    newdf.mesh = mydf.mesh
    has_eri = False
    if rank==0:
        has_eri = (mydf._cderi is not None)
    has_eri = comm.bcast(has_eri, root=0)
    if not has_eri:
        newdf.build()
        return newdf
    import h5py
    kptij_lst = None
    auxcell = df.df.make_modrho_basis(mydf.cell, mydf.auxbasis, mydf.exp_to_discard)
    nao, naux = mydf.cell.nao_nr(), auxcell.nao_nr()
    if rank==0:
        feri = h5py.File(mydf._cderi,'r')
        kptij_lst = numpy.asarray(feri['j3c-kptij'])
    else:
        feri = None
    kptij_lst = comm.bcast(kptij_lst, root=0)
    newdf.auxcell = auxcell
    newdf.kptij_lst = kptij_lst
    j3c = ctf.zeros([len(kptij_lst), nao, nao, naux], dtype=numpy.complex128)
    for i in range(len(kptij_lst)):
        if rank==0:
            idx_j3c = numpy.arange(nao**2*naux)
            tmp = numpy.asarray([feri['j3c/%d/%d'%(i,istep)] for istep in range(len(feri['j3c/%d'%i]))])[0]
            kpti_kptj = kptij_lst[i]
            is_real = is_zero(kpti_kptj[0]-kpti_kptj[1])
            if is_real:
                tmp = lib.unpack_tril(tmp)
            else:
                tmp = tmp.reshape(naux,nao,nao)
            tmp = tmp.transpose(1,2,0)
            j3c.write(i*idx_j3c.size+idx_j3c, tmp.ravel())
        else:
            j3c.write([],[])
    newdf.j3c = j3c
    return newdf

def _ao2mo_j3c(mydf, mo_coeff, nocc):
    '''
    ao2mo on density-fitted j3c integrals
    returns:
      ijL,     iaL,     aiL,    abL
    (ij|L),  (ia|L),  (ai|L), (ab|L)
    '''
    from pyscf.ctfcc.integrals import mpigdf
    nmo = mo_coeff.shape[-1]
    nvir = nmo - nocc
    if getattr(mydf, 'j3c', None) is None: mydf.build()
    kpts = mydf.kpts
    nkpts = len(kpts)
    nao, naux = mydf.j3c.shape[2:]
    ijL = ctf.zeros([nkpts,nkpts,nocc,nocc,naux], dtype=mydf.j3c.dtype)
    iaL = ctf.zeros([nkpts,nkpts,nocc,nvir,naux], dtype=mydf.j3c.dtype)
    aiL = ctf.zeros([nkpts,nkpts,nvir,nocc,naux], dtype=mydf.j3c.dtype)
    abL = ctf.zeros([nkpts,nkpts,nvir,nvir,naux], dtype=mydf.j3c.dtype)
    jobs = []
    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            jobs.append([ki,kj])
    tasks, ntasks = ctf_helper.static_partition(jobs)
    idx_j3c = numpy.arange(nao**2*naux)
    idx_ooL = numpy.arange(nocc**2*naux)
    idx_ovL = numpy.arange(nocc*nvir*naux)
    idx_vvL = numpy.arange(nvir**2*naux)
    cput1 = cput0 = (time.clock(), time.time())
    for itask in range(ntasks):
        if itask >= len(tasks):
            mydf.j3c.read([])
            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])

            ijL.write([], [])
            iaL.write([], [])
            aiL.write([], [])
            abL.write([], [])
            continue
        ki, kj = tasks[itask]
        ijid, ijdagger = mpigdf.get_member(kpts[ki], kpts[kj], mydf.kptij_lst)
        uvL = mydf.j3c.read(ijid*idx_j3c.size+idx_j3c).reshape(nao,nao,naux)
        if ijdagger: uvL = uvL.transpose(1,0,2).conj()
        pvL = numpy.einsum("up,uvL->pvL", mo_coeff[ki].conj(), uvL, optimize=True)
        uvL = None
        pqL = numpy.einsum('vq,pvL->pqL', mo_coeff[kj], pvL, optimize=True)

        off = ki * nkpts + kj
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

        off = kj * nkpts + ki
        pqL = pqL.transpose(1,0,2).conj()
        ijL.write(off*idx_ooL.size+idx_ooL, pqL[:nocc,:nocc].ravel())
        iaL.write(off*idx_ovL.size+idx_ovL, pqL[:nocc,nocc:].ravel())
        aiL.write(off*idx_ovL.size+idx_ovL, pqL[nocc:,:nocc].ravel())
        abL.write(off*idx_vvL.size+idx_vvL, pqL[nocc:,nocc:].ravel())

    return ijL, iaL, aiL, abL

def dump_to_file(mydf, cderi_file=None):
    """
    dump the eri from mpigdf.GDF object to file
    only one process used for writing to disk
    """
    if cderi_file is None:
        import tempfile
        if rank==0:
            cderi_file = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR).name
        cderi_file = comm.bcast(cderi_file, root=0)
    import h5py
    if rank==0:
        feri = h5py.File(cderi_file,'w')
        feri['j3c-kptij'] = mydf.kptij_lst
    else:
        feri = None
    nao, naux = mydf.j3c.shape[-2:]
    idx_j3c = numpy.arange(mydf.j3c.size).reshape(mydf.j3c.shape)
    for i in range(len(mydf.kptij_lst)):
        idx = idx_j3c[i].ravel()
        if rank!=0:
            tmp = mydf.j3c.read([])
        else:
            kpti_kptj = mydf.kptij_lst[i]
            is_real = is_zero(kpti_kptj[0]-kpti_kptj[1])
            tmp = mydf.j3c.read(idx).reshape(nao,nao,naux)
            if is_real:
                tmp = lib.pack_tril(tmp, axis=0).T
                if is_zero(kpti_kptj[0]):
                    tmp = tmp.real
            else:
                tmp = tmp.reshape(-1,naux).T
            feri['j3c/%d/0'%i] = tmp

    if rank==0:
        feri.close()
    mydf._cderi = cderi_file
    comm.barrier()
    return mydf

def _make_j3c(mydf, cell, auxcell, kptij_lst):
    '''
    j3c_junk integrals generated by slicing the auxilary basis
    density_fitted j3c integrals computed by slicing the kpoints
    '''
    max_memory = max(4000, mydf.max_memory-lib.current_memory()[0])
    fused_cell, fuse = df.df.fuse_auxcell(mydf, auxcell)
    nao, nfao = cell.nao_nr(), fused_cell.nao_nr()
    jobs = numpy.arange(fused_cell.nbas)
    tasks, ntasks = ctf_helper.static_partition(jobs)
    ntasks = max(comm.allgather(len(tasks)))
    j3c_junk = ctf.zeros([len(kptij_lst), nao**2, nfao], dtype=numpy.complex128)
    t1 =  t0 = (time.clock(), time.time())

    idx_full = numpy.arange(j3c_junk.size).reshape(j3c_junk.shape)
    if len(tasks) > 0:
        q0, q1 = tasks[0], tasks[-1] + 1
        shls_slice = (0, cell.nbas, 0, cell.nbas, q0, q1)
        bstart, bend = fused_cell.ao_loc_nr()[q0], fused_cell.ao_loc_nr()[q1]
        idx = idx_full[:,:,bstart:bend].ravel()
        tmp = df.incore.aux_e2(cell, fused_cell, intor='int3c2e', aosym='s2', kptij_lst=kptij_lst, shls_slice=shls_slice)
        nao_pair = nao**2
        if tmp.shape[-2] != nao_pair and tmp.ndim == 2:
            tmp = lib.unpack_tril(tmp, axis=0).reshape(nao_pair,-1)
        j3c_junk.write(idx, tmp.ravel())
    else:
        j3c_junk.write([],[])
    
    t1 = logger.timer(mydf, 'j3c_junk', *t1)

    naux = auxcell.nao_nr()
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    mydf.kptij_lst = kptij_lst
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    jobs = numpy.arange(len(uniq_kpts))
    tasks, ntasks = ctf_helper.static_partition(jobs)

    blksize = max(2048, int(max_memory*.5e6/16/fused_cell.nao_nr()))
    logger.debug2(mydf, 'max_memory %s (MB)  blocksize %s', max_memory, blksize)
    j2c  = ctf.zeros([len(uniq_kpts),naux,naux], dtype=numpy.complex128)

    a = cell.lattice_vectors() / (2*numpy.pi)
    def kconserve_indices(kpt):
        '''search which (kpts+kpt) satisfies momentum conservation'''
        kdif = numpy.einsum('wx,ix->wi', a, uniq_kpts + kpt)
        kdif_int = numpy.rint(kdif)
        mask = numpy.einsum('wi->i', abs(kdif - kdif_int)) < KPT_DIFF_TOL
        uniq_kptji_ids = numpy.where(mask)[0]
        return uniq_kptji_ids

    def cholesky_decomposed_metric(j2c_kptij):
        j2c_negative = None
        try:
            j2c_kptij = scipy.linalg.cholesky(j2c_kptij, lower=True)
            j2ctag = 'CD'
        except scipy.linalg.LinAlgError as e:
            w, v = scipy.linalg.eigh(j2c_kptij)
            logger.debug(mydf, 'cond = %.4g, drop %d bfns',
                      w[-1]/w[0], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = numpy.zeros(v.T.shape, dtype=v.dtype)
            v1[w>mydf.linear_dep_threshold,:] = v[:,w>mydf.linear_dep_threshold].conj().T
            v1[w>mydf.linear_dep_threshold,:] /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            j2c_kptij = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = numpy.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    j2c_negative = numpy.zeros(v1.shape, dtype=v1.dtype)
                    j2c_negative[idx,:] = (v[:,idx]/numpy.sqrt(-w[idx])).conj().T

            w = v = None
            j2ctag = 'eig'
        return j2c_kptij, j2c_negative, j2ctag

    for itask in range(ntasks):
        if itask >= len(tasks):
            j2c.write([],[])
            continue
        k = tasks[itask]
        kpt = uniq_kpts[k]
        j2ctmp = numpy.asarray(fused_cell.pbc_intor('int2c2e', hermi=1, kpts=kpt))
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            if is_zero(kpt):
                j2ctmp[naux:] -= numpy.dot(aoaux[naux:].conj()*coulG[p0:p1].conj(), aoaux.T).real
                j2ctmp[:naux,naux:] = j2ctmp[naux:,:naux].T
            else:
                j2ctmp[naux:] -= numpy.dot(aoaux[naux:].conj()*coulG[p0:p1].conj(), aoaux.T)
                j2ctmp[:naux,naux:] = j2ctmp[naux:,:naux].T.conj()

        tmp = fuse(fuse(j2ctmp).T).T
        idx = k * naux**2 + numpy.arange(naux**2)
        j2c.write(idx, tmp.ravel())
        j2ctmp = tmp = None

    coulG = None
    t1 = logger.timer(mydf, 'j2c', *t1)

    j3c = ctf.zeros([len(kpt_ji),nao,nao,naux], dtype=numpy.complex128)
    jobs = numpy.arange(len(kpt_ji))
    tasks, ntasks = ctf_helper.static_partition(jobs)

    for itask in range(ntasks):
        if itask >= len(tasks):
            j2c_ji = j2c.read([])
            j3ctmp = j3c_junk.read([])
            j3c.write([],[])
            continue
        idx_ji = tasks[itask]
        kpti, kptj = kptij_lst[idx_ji]
        idxi, idxj = member(kpti, mydf.kpts), member(kptj, mydf.kpts)
        uniq_idx = uniq_inverse[idx_ji]
        kpt = uniq_kpts[uniq_idx]
        id_eq = kconserve_indices(-kpt)
        id_conj = kconserve_indices(kpt)
        id_conj = numpy.asarray([i for i in id_conj if i not in id_eq], dtype=int)
        id_full = numpy.hstack((id_eq, id_conj))
        map_id, conj = min(id_full), numpy.argmin(id_full) >=len(id_eq)
        j2cidx = map_id * naux**2 + numpy.arange(naux**2)
        j2c_ji = j2c.read(j2cidx).reshape(naux, naux) # read to be added
        j2c_ji, j2c_negative, j2ctag = cholesky_decomposed_metric(j2c_ji)
        if conj: j2c_ji = j2c_ji.conj()
        shls_slice= (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        wcoulG = mydf.weighted_coulG(kpt, False, mesh)
        Gaux *= wcoulG.reshape(-1,1)
        j3c_id = idx_ji * nao**2*nfao + numpy.arange(nao**2*nfao)
        j3ctmp = j3c_junk.read(j3c_id).reshape(nao**2, fused_cell.nao_nr()).T
        if is_zero(kpt):  # kpti == kptj
            if cell.dimension == 3:
                vbar = fuse(mydf.auxbar(fused_cell))
                ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kptj)
                for i in numpy.where(vbar != 0)[0]:
                    j3ctmp[i] -= vbar[i] * ovlp.reshape(-1)

        aoao = ft_ao._ft_aopair_kpts(cell, Gv, None, 's1', b, gxyz, Gvbase, kpt, kptj)[0].reshape(len(Gv),-1)
        j3ctmp[naux:] -= numpy.dot(Gaux.T.conj(), aoao)
        j3ctmp = fuse(j3ctmp)
        if j2ctag == 'CD':
            v = scipy.linalg.solve_triangular(j2c_ji, j3ctmp, lower=True, overwrite_b=True)
        else:
            v = numpy.dot(j2c_ji, j3ctmp)
        v = v.T.reshape(nao,nao,naux)
        j3c_id = idx_ji * nao**2*naux + numpy.arange(nao**2*naux)
        j3c.write(j3c_id, v.ravel())

    mydf.j3c = j3c
    return None


class GDF(df.GDF):
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.verbose = cell.verbose
        self.j3c = None
        self.kptij_lst = None
        df.GDF.__init__(self, cell, kpts)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if self.kpts_band is not None:
            self.kpts_band = numpy.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(numpy.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = df.df.make_modrho_basis(self.cell, self.auxbasis,
                                         self.exp_to_discard)

        if self.kpts_band is None:
            kpts = self.kpts
            kband_uniq = numpy.zeros((0,3))
        else:
            kpts = self.kpts
            kband_uniq = [k for k in self.kpts_band if len(member(k, kpts))==0]
        if j_only is None:
            j_only = self._j_only
        if j_only:
            kall = numpy.vstack([kpts,kband_uniq])
            kptij_lst = numpy.hstack((kall,kall)).reshape(-1,2,3)
        else:
            kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
            kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
            kptij_lst.extend([(ki, ki) for ki in kband_uniq])
            kptij_lst = numpy.asarray(kptij_lst)
        t1 = (time.clock(), time.time())
        self._make_j3c(self.cell, self.auxcell, kptij_lst)
        t1 = logger.timer(self, 'j3c', *t1)
        return self

    dump_to_file = dump_to_file
    from_serial = from_serial
    _make_j3c = _make_j3c
    _ao2mo_j3c = _ao2mo_j3c

if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    cell = gto.Cell()
    cell.atom='''
    H 0.000000000000   0.000000000000   0.000000000000
    H 1.685068664391   1.685068664391   1.685068664391
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
    mydf = GDF(cell, kpts)
    mydf.mesh = [5,5,5]
    mydf.build()
    mydf.dump_to_file()
    if rank==0:
        mf= scf.KRHF(cell,kpts)
        mf.with_df = mydf
        #mf.with_df = df.GDF(cell,kpts) # USING SERIAL GDF ERI
        #mf.with_df._cderi = mydf._cderi
        #mf.with_df.mesh = [5,5,5]
        #mf.with_df = mydf
        #print(mydf.__dict__.keys(), mf.with_df.__dict__.keys())
        #print(mydf._cderi, mf.with_df._cderi)
        e1 =mf.kernel()
        print(ehf - -1.3606944021652319)
