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

'''core module for CC/k-CC ao2mo transformation'''
import numpy
import time
from pyscf import gto, ao2mo, lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from symtensor.ctf import einsum, frombatchfunc, array
import pyscf.pbc.tools.pbc as tools
import itertools

def _make_ao_ints(mol, mo_coeff, nocc, shls_slice):
    nao, nmo = mo_coeff.shape
    ao_loc = mol.ao_loc_nr()
    intor = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    nvir = nmo - nocc
    sqidx = numpy.arange(nao**2).reshape(nao,nao)
    ish0, ish1, jsh0, jsh1 = shls_slice
    i0, i1 = ao_loc[ish0], ao_loc[ish1]
    j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
    di = i1 - i0
    dj = j1 - j0
    if i0 != j0:
        eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                      shls_slice=shls_slice, aosym='s2kl',
                                      ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
        eri = _ao2mo.nr_e2(eri.reshape(di*dj,-1), mo_coeff, (0,nmo,0,nmo), 's2kl', 's1')
        idxij = sqidx[i0:i1,j0:j1].ravel()

        eri = eri.reshape(-1,nmo,nmo)
        eri_ji = eri.reshape(di,dj,-1).transpose(1,0,2).reshape(-1,nmo,nmo)

        idxji = sqidx[j0:j1,i0:i1].ravel()
        idxoo_ij = (idxij[:,None] * nocc**2 + numpy.arange(nocc**2)).ravel()
        idxov_ij = (idxij[:,None] * nocc*nvir + numpy.arange(nocc*nvir)).ravel()
        idxvv_ij = (idxij[:,None] * nvir**2 + numpy.arange(nvir**2)).ravel()

        idxoo_ji = (idxji[:,None] * nocc**2 + numpy.arange(nocc**2)).ravel()
        idxov_ji = (idxji[:,None] * nocc*nvir + numpy.arange(nocc*nvir)).ravel()
        idxvv_ji = (idxji[:,None] * nvir**2 + numpy.arange(nvir**2)).ravel()

        idxoo = numpy.concatenate([idxoo_ij, idxoo_ji])
        idxov = numpy.concatenate([idxov_ij, idxov_ji])
        idxvv = numpy.concatenate([idxvv_ij, idxvv_ji])

        ppoo = numpy.concatenate([eri[:,:nocc,:nocc].ravel(), eri_ji[:,:nocc,:nocc].ravel()])
        ppov = numpy.concatenate([eri[:,:nocc,nocc:].ravel(), eri_ji[:,:nocc,nocc:].ravel()])
        ppvv = numpy.concatenate([eri[:,nocc:,nocc:].ravel(), eri_ji[:,nocc:,nocc:].ravel()])

    else:
        eri = gto.moleintor.getints4c(intor, mol._atm, mol._bas, mol._env,
                                      shls_slice=shls_slice, aosym='s4',
                                      ao_loc=ao_loc, cintopt=ao2mopt._cintopt)
        eri = _ao2mo.nr_e2(eri, mo_coeff, (0,nmo,0,nmo), 's4', 's1')

        idx = sqidx[i0:i1,j0:j1].ravel()

        idxoo = (idx[:,None] * nocc**2 + numpy.arange(nocc**2)).ravel()
        idxov = (idx[:,None] * nocc*nvir + numpy.arange(nocc*nvir)).ravel()
        idxvv = (idx[:,None] * nvir**2 + numpy.arange(nvir**2)).ravel()

        eri = lib.unpack_tril(eri, axis=0).reshape(-1,nmo,nmo)
        ppoo = eri[:,:nocc,:nocc].ravel()
        ppov = eri[:,:nocc,nocc:].ravel()
        ppvv = eri[:,nocc:,nocc:].ravel()

    return (idxoo, idxov, idxvv), (ppoo, ppov, ppvv)

def make_ao_ints(mol, mo_coeff, nocc):
    '''
    partial ao2mo transformation, complex mo_coeff supported
    returns:
      ppoo,     ppov,     ppvv
    (uv|ij),  (uv|ia),  (uv|ab)
    '''
    ao_loc = mol.ao_loc_nr()
    mo = numpy.asarray(mo_coeff, order='F')
    nao, nmo = mo.shape
    nvir = nmo - nocc
    dtype = mo.dtype
    blksize = int(max(4, min(nao/3, 2000e6/8/nao**3)))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize)
    tasks = []
    for k, (ish0, ish1, di) in enumerate(sh_ranges):
        for jsh0, jsh1, dj in sh_ranges[:k+1]:
            tasks.append((ish0,ish1,jsh0,jsh1))

    shape_list = [(nao,nao,nocc,nocc),\
                  (nao,nao,nocc,nvir),\
                  (nao,nao,nvir,nvir)]

    make_ao_eri = lambda *shls_slice: _make_ao_ints(mol, mo, nocc, shls_slice)
    ppoo, ppov, ppvv  = frombatchfunc(make_ao_eri, shape_list, tasks, dtype=dtype, nout=3)
    return ppoo.array, ppov.array, ppvv.array

def _make_fftdf_j3c(mydf, ki, kj, mo_a, mo_b):
    cell = mydf.cell
    nao = cell.nao_nr()
    coords = cell.gen_uniform_grids(mydf.mesh)
    if mo_a.shape==mo_b.shape:
        RESTRICTED = numpy.linalg.norm(mo_a-mo_b) < 1e-10
    else:
        RESTRICTED = False
    ngrids = len(coords)
    nmoa, nmob = mo_a.shape[-1], mo_b.shape[-1]

    idx_ppG = numpy.arange(nmoa**2*ngrids)
    idx_ppR = numpy.arange(nmob**2*ngrids)

    kpts = mydf.kpts
    nkpts = len(kpts)
    kpti, kptj = kpts[ki], kpts[kj]
    ao_kpti = mydf._numint.eval_ao(cell, coords, kpti)[0]
    ao_kptj = mydf._numint.eval_ao(cell, coords, kptj)[0]
    q = kptj - kpti
    coulG = tools.get_coulG(cell, q, mesh=mydf.mesh)
    wcoulG = coulG * (cell.vol/ngrids)
    fac = numpy.exp(-1j * numpy.dot(coords, q))
    mo_kpti_b = numpy.dot(ao_kpti, mo_b[ki]).T
    mo_kptj_b = numpy.dot(ao_kptj, mo_b[kj]).T
    mo_pairs_b = numpy.einsum('ig,jg->ijg', mo_kpti_b.conj(), mo_kptj_b)

    if RESTRICTED:
        mo_pairs_a = mo_pairs_b
    else:
        mo_kpti_a = numpy.dot(ao_kpti, mo_a[ki]).T
        mo_kptj_a = numpy.dot(ao_kptj, mo_a[kj]).T
        mo_pairs_a = numpy.einsum('ig,jg->ijg', mo_kpti_a.conj(), mo_kptj_a)

    mo_pairs_G = tools.fft(mo_pairs_a.reshape(-1,ngrids)*fac, mydf.mesh)

    if ki==kj:
        ind_ppR  = (ki*nkpts+kj)*idx_ppR.size+idx_ppR
        val_ppR = mo_pairs_b.ravel()
    else:
        ind_ppR = numpy.concatenate([(ki*nkpts+kj)*idx_ppR.size+idx_ppR,\
                                     (kj*nkpts+ki)*idx_ppR.size+idx_ppR])
        val_ppR = numpy.concatenate([mo_pairs_b.ravel(),\
                                     mo_pairs_b.transpose(1,0,2).conj().ravel()])

    mo_pairs_a = mo_pairs_b = None
    mo_pairs_G*= wcoulG

    v = tools.ifft(mo_pairs_G, mydf.mesh)
    v *= fac.conj()
    v = v.reshape(nmoa,nmoa,ngrids)

    if ki==kj:
        ind_ppG = (ki*nkpts+kj)*idx_ppG.size+idx_ppG
        val_ppG = v.ravel()
    else:
        ind_ppG = numpy.concatenate([(ki*nkpts+kj)*idx_ppG.size+idx_ppG,\
                                     (kj*nkpts+ki)*idx_ppG.size+idx_ppG])
        val_ppG = numpy.concatenate([v.ravel(),\
                                     v.transpose(1,0,2).conj().ravel()])
    return (ind_ppG, ind_ppR), (val_ppG, val_ppR)

def _make_fftdf_eris(mycc, mo_a, mo_b, nocca, noccb, out=None):
    mydf = mycc._scf.with_df
    kpts = mycc.kpts
    cell = mydf.cell
    gvec = cell.reciprocal_vectors()
    coords = cell.gen_uniform_grids(mydf.mesh)
    ngrids = len(coords)
    nkpts = len(kpts)
    nmoa, nmob = mo_a.shape[-1], mo_b.shape[-1]
    cput1 = cput0 = (time.clock(), time.time())
    all_tasks = []

    for ki in range(nkpts):
        for kj in range(ki,nkpts):
            all_tasks.append([ki,kj])

    sym1 = ["+-+", [kpts, kpts, kpts[0]-kpts], None, gvec]
    sym2 = ["+--", [kpts, kpts, kpts[0]-kpts], None, gvec]
    sym_list  = [sym1, sym2]

    gen_eri  = lambda ki, kj: _make_fftdf_j3c(mydf, ki, kj, mo_a, mo_b)
    shape_list = [(nmoa, nmoa, ngrids), (nmob, nmob, ngrids)]

    ppG, ppR = frombatchfunc(gen_eri, shape_list, all_tasks, \
                             sym=sym_list, nout=2, dtype=numpy.complex128)

    ooG = ppG[:,:,:nocca,:nocca]
    ovG = ppG[:,:,:nocca,nocca:]
    vvG = ppG[:,:,nocca:,nocca:]

    del ppG

    ooR = ppR[:,:,:noccb,:noccb]
    ovR = ppR[:,:,:noccb,noccb:]
    voR = ppR[:,:,noccb:,:noccb]
    vvR = ppR[:,:,noccb:,noccb:]

    del ppR
    oooo = einsum('ijg,klg->ijkl', ooG, ooR)/ nkpts
    ooov = einsum('ijg,kag->ijka', ooG, ovR)/ nkpts
    oovv = einsum('ijg,abg->ijab', ooG, vvR)/ nkpts
    ooG = ooR = ijG = ijR = None
    ovvo = einsum('iag,bjg->iabj', ovG, voR)/ nkpts
    ovov = einsum('iag,jbg->iajb', ovG, ovR)/ nkpts
    ovR = iaR = voR = aiR = None
    ovvv = einsum('iag,bcg->iabc', ovG, vvR)/ nkpts
    ovG = iaG = None
    vvvv = einsum('abg,cdg->abcd', vvG, vvR)/ nkpts

    cput1 = logger.timer(mycc, "(pq|G) to (pq|rs)", *cput1)

    if out is None:
        return oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv
    else:
        return oooo+out[0], ooov+out[1], oovv+out[2], \
               ovvo+out[3], ovov+out[4], ovvv+out[5], vvvv+out[6]

def make_fftdf_eris_rhf(mycc, eris):
    mo_coeff = eris.mo_coeff
    nocc, nmo = mycc.nocc, mycc.nmo
    nvir = nmo - nocc
    oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                _make_fftdf_eris(mycc, mo_coeff, mo_coeff, nocc, nocc)
    eris.oooo = oooo
    eris.ooov = ooov
    eris.oovv = oovv
    eris.ovvo = ovvo
    eris.ovov = ovov
    eris.ovvv = ovvv
    eris.vvvv = vvvv

def make_fftdf_eris_uhf(mycc, eris):
    mo_a, mo_b = eris.mo_coeff[0], eris.mo_coeff[1]
    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                _make_fftdf_eris(mycc, mo_a, mo_a, nocca, nocca)
    eris.oooo = oooo
    eris.ooov = ooov
    eris.oovv = oovv
    eris.ovov = ovov
    eris.voov = ovvo.transpose(1,0,3,2).conj()
    eris.vovv = ovvv.transpose(1,0,3,2).conj()
    eris.vvvv = vvvv

    OOOO, OOOV, OOVV, OVVO, OVOV, OVVV, VVVV = \
                _make_fftdf_eris(mycc, mo_b, mo_b, noccb, noccb)
    eris.OOOO = OOOO
    eris.OOOV = OOOV
    eris.OOVV = OOVV
    eris.OVOV = OVOV
    eris.VOOV = OVVO.transpose(1,0,3,2).conj()
    eris.VOVV = OVVV.transpose(1,0,3,2).conj()
    eris.VVVV = VVVV

    ooOO, ooOV, ooVV, ovVO, ovOV, ovVV, vvVV = \
                _make_fftdf_eris(mycc, mo_a, mo_b, nocca, noccb)
    eris.ooOO = ooOO
    eris.ooOV = ooOV
    eris.ooVV = ooVV
    eris.ovOV = ovOV
    eris.voOV = ovVO.transpose(1,0,3,2).conj()
    eris.voVV = ovVV.transpose(1,0,3,2).conj()
    eris.vvVV = vvVV

    _, OOov, OOvv, OVvo, OVov, OVvv, _ = \
                _make_fftdf_eris(mycc, mo_b, mo_a, noccb, nocca)
    eris.OOov = OOov
    eris.OOvv = OOvv
    eris.OVov = OVov
    eris.VOov = OVvo.transpose(1,0,3,2).conj()
    eris.VOvv = OVvv.transpose(1,0,3,2).conj()

def make_fftdf_eris_ghf(mycc, eris):
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    nkpts = mycc.nkpts
    nao = mycc._scf.cell.nao_nr()
    if getattr(eris.mo_coeff[0], 'orbspin', None) is None:
        # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
        mo_a_coeff = numpy.asarray([mo[:nao] for mo in eris.mo_coeff])
        mo_b_coeff = numpy.asarray([mo[nao:] for mo in eris.mo_coeff])
        eri = _make_fftdf_eris(mycc, mo_a_coeff, mo_a_coeff, nocc, nocc)
        eri = _make_fftdf_eris(mycc, mo_b_coeff, mo_b_coeff, nocc, nocc, eri)
        eri = _make_fftdf_eris(mycc, mo_a_coeff, mo_b_coeff, nocc, nocc, eri)
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv =\
                            _make_fftdf_eris(mycc, mo_b_coeff, mo_a_coeff, nocc, nocc, eri)
        eri = None
    else:
        mo_a_coeff = numpy.asarray([mo[:nao] + mo[nao:] for mo in eris.mo_coeff])
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                        _make_fftdf_eris(mycc, mo_a_coeff, mo_a_coeff, nocc, nocc)

        all_tasks = [[kp,kq,kr] for kp,kq,kr in itertools.product(range(nkpts), repeat=3)]
        orb_dic = {"o": nocc, "v":nvir}
        def _force_sym(kp, kq, kr):
            ks = eris.kconserv[kp,kq,kr]
            off = (kp*nkpts**2+kq*nkpts+kr)
            orb = [getattr(eris.mo_coeff[kx], 'orbspin') for kx in [kp, kq, kr, ks]]

            def get_idx(symbol):
                offset = off * numpy.prod([orb_dic[i] for i in symbol])
                pq_size = numpy.prod([orb_dic[i] for i in symbol[:2]])
                rs_size = numpy.prod([orb_dic[i] for i in symbol[2:]])
                slice_list = []
                orb_list = []
                for ni, i in enumerate(symbol):
                    if i=='o':
                        orb_list.append(orb[ni][:nocc])
                    else:
                        orb_list.append(orb[ni][nocc:])

                pqforbid = numpy.where((orb_list[0][:,None] != orb_list[1]).ravel())[0]
                rsforbid = numpy.where((orb_list[2][:,None] != orb_list[3]).ravel())[0]

                idx_pq = offset + pqforbid[:,None] * rs_size + numpy.arange(rs_size)
                idx_rs = offset + numpy.arange(rs_size)[:,None] * rs_size + rsforbid.ravel()
                idx=  numpy.concatenate((idx_pq.ravel(), idx_rs.ravel()))
                val = numpy.zeros(idx.size)
                return idx, val

            idx_oooo, val_oooo = get_idx('oooo')
            idx_ooov, val_ooov = get_idx('ooov')
            idx_oovv, val_oovv = get_idx('oovv')
            idx_ovvo, val_ovvo = get_idx('ovvo')
            idx_ovov, val_ovov = get_idx('ovov')
            idx_ovvv, val_ovvv = get_idx('ovvv')
            idx_vvvv, val_vvvv = get_idx('vvvv')

            return (idx_oooo, idx_ooov, idx_oovv, idx_ovvo, idx_ovov, idx_ovvv, idx_vvvv),\
                   (val_oooo, val_ooov, val_oovv, val_ovvo, val_ovov, val_ovvv, val_vvvv)

        out = (oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv)
        shape_list = [i.shape for i in out]
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                    frombatchfunc(_force_sym, shape_list, all_tasks, out=out)

    eris.vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(2,0,1,3)
    eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    del vvvv, ovvv
    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(2,0,1,3)
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    eris.ovov = oovv.transpose(0,2,1,3) - ovvo.transpose(0,2,3,1)
    eris.ovvo = ovvo.transpose(0,2,1,3) - oovv.transpose(0,2,3,1)
    del oooo, ooov, ovov, oovv, ovvo

def make_df_eris_rhf(mycc, eris):
    mydf = mycc._scf.with_df
    mo_coeff = eris.mo_coeff
    nocc = mycc.nocc
    kpts = mycc.kpts
    nkpts = len(kpts)
    gvec = mydf.cell.reciprocal_vectors()
    cput1 = (time.clock(), time.time())
    ijL, iaL, aiL, abL = mydf._ao2mo_j3c(mo_coeff, nocc)
    cput1 = logger.timer(mycc, "j3c transformation", *cput1)
    sym1 = ["+-+", [kpts, kpts, kpts[0]-kpts], None, gvec]
    sym2 = ["+--", [kpts, kpts, kpts[0]-kpts], None, gvec]

    ooL = array(ijL, sym1)
    ovL = array(iaL, sym1)
    voL = array(aiL, sym1)
    vvL = array(abL, sym1)

    ooL2 = array(ijL, sym2)
    ovL2 = array(iaL, sym2)
    voL2 = array(aiL, sym2)
    vvL2 = array(abL, sym2)

    eris.oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
    eris.ooov = einsum('ijg,kag->ijka', ooL, ovL2) / nkpts
    eris.oovv = einsum('ijg,abg->ijab', ooL, vvL2) / nkpts
    eris.ovvo = einsum('iag,bjg->iabj', ovL, voL2) / nkpts
    eris.ovov = einsum('iag,jbg->iajb', ovL, ovL2) / nkpts
    eris.ovvv = einsum('iag,bcg->iabc', ovL, vvL2) / nkpts
    eris.vvvv = einsum('abg,cdg->abcd', vvL, vvL2) / nkpts

    cput1 = logger.timer(mycc, "integral transformation", *cput1)

def make_df_eris_uhf(mycc, eris):
    mydf = mycc._scf.with_df
    mo_a, mo_b = eris.mo_coeff[0], eris.mo_coeff[1]
    nocca, noccb = mycc.nocc
    kpts = mycc.kpts
    nkpts = len(kpts)
    gvec = mydf.cell.reciprocal_vectors()
    cput1 = (time.clock(), time.time())

    ijL, iaL, aiL, abL = mydf._ao2mo_j3c(mo_a, nocca)
    IJL, IAL, AIL, ABL = mydf._ao2mo_j3c(mo_b, noccb)
    cput1 = logger.timer(mycc, "(uv|L)->(pq|L)", *cput1)

    sym1 = ["+-+", [kpts, kpts, kpts[0]-kpts], None, gvec]
    sym2 = ["+--", [kpts, kpts, kpts[0]-kpts], None, gvec]

    ooL = array(ijL, sym1)
    ovL = array(iaL, sym1)
    voL = array(aiL, sym1)
    vvL = array(abL, sym1)

    ooL2 = array(ijL, sym2)
    ovL2 = array(iaL, sym2)
    voL2 = array(aiL, sym2)
    vvL2 = array(abL, sym2)

    OOL = array(IJL, sym1)
    OVL = array(IAL, sym1)
    VOL = array(AIL, sym1)
    VVL = array(ABL, sym1)

    OOL2 = array(IJL, sym2)
    OVL2 = array(IAL, sym2)
    VOL2 = array(AIL, sym2)
    VVL2 = array(ABL, sym2)

    eris.oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
    eris.ooov = einsum('ijg,kag->ijka', ooL, ovL2) / nkpts
    eris.oovv = einsum('ijg,abg->ijab', ooL, vvL2) / nkpts
    eris.voov = einsum('iag,jbg->aibj', ovL, voL2).conj() /nkpts
    eris.ovov = einsum('iag,jbg->iajb', ovL, ovL2) / nkpts
    eris.vovv = einsum('iag,bcg->aicb', ovL, vvL2).conj() / nkpts
    eris.vvvv = einsum('abg,cdg->abcd', vvL, vvL2) / nkpts

    eris.OOOO = einsum('ijg,klg->ijkl', OOL, OOL2) / nkpts
    eris.OOOV = einsum('ijg,kag->ijka', OOL, OVL2) / nkpts
    eris.OOVV = einsum('ijg,abg->ijab', OOL, VVL2) / nkpts
    eris.VOOV = einsum('iag,jbg->aibj', OVL, VOL2).conj() /nkpts
    eris.OVOV = einsum('iag,jbg->iajb', OVL, OVL2) / nkpts
    eris.VOVV = einsum('iag,bcg->aicb', OVL, VVL2).conj() / nkpts
    eris.VVVV = einsum('abg,cdg->abcd', VVL, VVL2) / nkpts

    eris.ooOO = einsum('ijg,klg->ijkl', ooL, OOL2) / nkpts
    eris.ooOV = einsum('ijg,kag->ijka', ooL, OVL2) / nkpts
    eris.ooVV = einsum('ijg,abg->ijab', ooL, VVL2) / nkpts
    eris.ovOV = einsum('iag,jbg->iajb', ovL, OVL2) / nkpts
    eris.voOV = einsum('iag,jbg->aibj', ovL, VOL2).conj() /nkpts
    eris.voVV = einsum('iag,bcg->aicb', ovL, VVL2).conj() / nkpts
    eris.vvVV = einsum('abg,cdg->abcd', vvL, VVL2) / nkpts

    eris.OOov = einsum('ijg,kag->ijka', OOL, ovL2) / nkpts
    eris.OOvv = einsum('ijg,abg->ijab', OOL, vvL2) / nkpts
    eris.OVov = einsum('iag,jbg->iajb', OVL, ovL2) / nkpts
    eris.VOov = einsum('iag,jbg->aibj', OVL, voL2).conj() /nkpts
    eris.VOvv = einsum('iag,bcg->aicb', OVL, vvL2).conj() / nkpts

    cput1 = logger.timer(mycc, "integral transformation", *cput1)

def make_df_eris_ghf(mycc, eris):
    nocc = mycc.nocc
    nkpts = mycc.nkpts
    nao = mycc._scf.cell.nao_nr()
    nocc = mycc.nocc
    mydf = mycc._scf.with_df
    if getattr(eris.mo_coeff[0], 'orbspin', None) is None:
        # The bottom nao//2 coefficients are down (up) spin while the top are up (down).
        mo_a_coeff = numpy.asarray([mo[:nao] for mo in eris.mo_coeff])
        mo_b_coeff = numpy.asarray([mo[nao:] for mo in eris.mo_coeff])

        ijL, iaL, aiL, abL = mydf._ao2mo_j3c(mo_a_coeff, nocc)
        IJL, IAL, AIL, ABL = mydf._ao2mo_j3c(mo_b_coeff, nocc)
        ijL += IJL
        iaL += IAL
        aiL += AIL
        abL += ABL
        del IJL, IAL, AIL, ABL

        ooL = array(ijL, sym1)
        ovL = array(iaL, sym1)
        voL = array(aiL, sym1)
        vvL = array(abL, sym1)

        ooL2 = array(ijL, sym2)
        ovL2 = array(iaL, sym2)
        voL2 = array(aiL, sym2)
        vvL2 = array(abL, sym2)

        oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
        ooov = einsum('ijg,klg->ijkl', ooL, ovL2) / nkpts
        oovv = einsum('ijg,klg->ijkl', ooL, vvL2) / nkpts
        ovvo = einsum('ijg,klg->ijkl', ovL, voL2) / nkpts
        ovov = einsum('ijg,klg->ijkl', ovL, ovL2) / nkpts
        ovvv = einsum('ijg,klg->ijkl', ovL, vvL2) / nkpts
        vvvv = einsum('ijg,klg->ijkl', vvL, vvL2) / nkpts

        del ooL, ovL, voL, vvL, ooL2, ovL2, voL2, vvL2, ijL, iaL, aiL, abL
    else:
        mo_a_coeff = numpy.asarray([mo[:nao] + mo[nao:] for mo in eris.mo_coeff])
        ijL, iaL, aiL, abL = mydf._ao2mo_j3c(mo_a_coeff, nocc)

        ooL = array(ijL, sym1)
        ovL = array(iaL, sym1)
        voL = array(aiL, sym1)
        vvL = array(abL, sym1)

        ooL2 = array(ijL, sym2)
        ovL2 = array(iaL, sym2)
        voL2 = array(aiL, sym2)
        vvL2 = array(abL, sym2)

        oooo = einsum('ijg,klg->ijkl', ooL, ooL2) / nkpts
        ooov = einsum('ijg,klg->ijkl', ooL, ovL2) / nkpts
        oovv = einsum('ijg,klg->ijkl', ooL, vvL2) / nkpts
        ovvo = einsum('ijg,klg->ijkl', ovL, voL2) / nkpts
        ovov = einsum('ijg,klg->ijkl', ovL, ovL2) / nkpts
        ovvv = einsum('ijg,klg->ijkl', ovL, vvL2) / nkpts
        vvvv = einsum('ijg,klg->ijkl', vvL, vvL2) / nkpts

        all_tasks = [[kp,kq,kr] for kp,kq,kr in itertools.product(range(nkpts), repeat=3)]
        orb_dic = {"o": nocc, "v":nvir}
        def _force_sym(kp, kq, kr):
            ks = eris.kconserv[kp,kq,kr]
            off = (kp*nkpts**2+kq*nkpts+kr)
            orb = [getattr(eris.mo_coeff[kx], 'orbspin') for kx in [kp, kq, kr, ks]]
            pqidx = numpy.unravel_index(numpy.where((orb[0][:,None] != orb[1]).ravel()), (2*nao,2*nao))
            rsidx = numpy.unravel_index(numpy.where((orb[2][:,None] != orb[3]).ravel()), (2*nao,2*nao))

            def get_idx(symbol):
                offset = off * numpy.prod([orb_dic[i] for i in symbol])
                pq_size = numpy.prod([orb_dic[i] for i in symbol[:2]])
                rs_size = numpy.prod([orb_dic[i] for i in symbol[2:]])
                slice_list = []
                for i in symbol:
                    if i=='o':
                        slice_list.append(slice(None,nocc,None))
                    else:
                        slice_list.append(slice(nocc,None,None))
                slice_pq = tuple(slice_list[:2])
                slice_rs = tuple(slice_list[2:])
                idx_pq = offset + pqidx[slice_pq].ravel()[:,None] * rs_size + numpy.arange(rs_size)
                idx_rs = offset + numpy.arange(rs_size)[:,None] * rs_size + rsidx[slice_rs].ravel()
                idx=  numpy.concatenate((idx_pq, idx_rs))
                val = numpy.zeros(idx.size)
                return idx, val

            idx_oooo, val_oooo = get_idx('oooo')
            idx_ooov, val_ooov = get_idx('ooov')
            idx_oovv, val_oovv = get_idx('oovv')
            idx_ovvo, val_ovvo = get_idx('ovvo')
            idx_ovov, val_ovov = get_idx('ovov')
            idx_ovvv, val_ovvv = get_idx('ovvv')
            idx_vvvv, val_vvvv = get_idx('vvvv')

            return (idx_oooo, idx_ooov, idx_oovv, idx_ovvo, idx_ovov, idx_ovvv, idx_vvvv),\
                   (val_oooo, val_ooov, val_oovv, val_ovvo, val_ovov, val_ovvv, val_vvvv)

        out = (oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv)
        shape_list = [i.shape for i in out]
        oooo, ooov, oovv, ovvo, ovov, ovvv, vvvv = \
                    frombatchfunc(_force_sym, shape_list, all_tasks, out=out)

    eris.vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(2,0,1,3)
    eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    del vvvv, ovvv
    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(2,0,1,3)
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    eris.ovov = oovv.transpose(0,2,1,3) - ovvo.transpose(0,2,3,1)
    eris.ovvo = ovvo.transpose(0,2,1,3) - oovv.transpose(0,2,3,1)
    del oooo, ooov, ovov, oovv, ovvo
