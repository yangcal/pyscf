#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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

import itertools
import numpy as np
import time
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kump2 import padding_k_idx
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.ctfcc import ctf_helper
from symtensor.ctf import array, einsum, frombatchfunc
from symtensor.ctf.backend import asarray


def kernel(mycc, eris, t1=None, t2=None, slice_size=None):
    cpu1 = cpu0 = (time.clock(), time.time())
    if t1 is None or t2 is None:
        t1, t2 = mycc.t1, mycc.t2

    log = logger.Logger(mycc.stdout, mycc.verbose)

    nkpts = mycc.nkpts
    kpts= mycc.kpts
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    if slice_size is None:
        slice_size = nkpts**3*max(nocca, noccb)*max(nvira, nvirb)**3
    else:
        slice_size = int(slice_size /6. * 6.25e4)
    mo_ea_o = [e[:nocca] for e in eris.mo_energy[0]]
    mo_eb_o = [e[:noccb] for e in eris.mo_energy[1]]
    mo_ea_v = [e[nocca:] + mycc.level_shift for e in eris.mo_energy[0]]
    mo_eb_v = [e[noccb:] + mycc.level_shift for e in eris.mo_energy[1]]

    nonzero_padding_a, nonzero_padding_b = padding_k_idx(mycc, kind="split")
    nonzero_opadding_a, nonzero_vpadding_a = nonzero_padding_a
    nonzero_opadding_b, nonzero_vpadding_b = nonzero_padding_b
    mo_oa_script = (0, nocca, mo_ea_o, nonzero_opadding_a)
    mo_ob_script = (0, noccb, mo_eb_o, nonzero_opadding_b)
    mo_va_script = (0, nvira, mo_ea_v, nonzero_vpadding_a)
    mo_vb_script = (0, nvirb, mo_eb_v, nonzero_vpadding_b)

    fvo = eris.fov.transpose(1,0).conj()
    fVO = eris.fOV.transpose(1,0).conj()

    def r6(w):
        return (w + w.transpose(0,1,2,5,3,4) + w.transpose(0,1,2,4,5,3) \
                - w.transpose(0,1,2,5,4,3) - w.transpose(0,1,2,3,5,4) \
                - w.transpose(0,1,2,4,3,5))

    def get_w_s6(ka,kb,kc,orbslice,t2t,vovv,vooo):
        a0,a1,b0,b1,c0,c1=orbslice
        ttmp = t2t[ka,:,:,a0:a1]
        wtmp = vovv[kc,:,kb,c0:c1,:,b0:b1]
        w = einsum('aeij,ckbe->abcijk', ttmp, wtmp)
        ttmp = t2t[kb,kc,:,b0:b1,c0:c1]
        wtmp = vooo[ka,:,:,a0:a1]
        w-= einsum('bcmk,aimj->abcijk', ttmp, wtmp)
        return w

    def get_v_s6(ka,kb,kc,orbslice,t1t,t2t,vvoo,vo):
        a0,a1,b0,b1,c0,c1=orbslice
        kia = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[ka]))
        ttmp = einsum('Iia,I->Iia',t1t[:,:,a0:a1].array,kia)
        wtmp = vvoo[kb,kc,:,b0:b1,c0:c1].array
        v = einsum('Jbcjk,Iia->IJabcijk', wtmp, ttmp)
        ttmp = t2t[kb,kc,:,b0:b1,c0:c1].array
        wtmp = einsum('Iai,I->Iai', vo[:,a0:a1], kia)
        v+= einsum('Jbcjk,Iai->IJabcijk', ttmp, wtmp) * .5
        return array(v, mycc.gen_sym("000+++", kpts[ka]+kpts[kb]+kpts[kc]))

    def get_eijk(ka,kb,kc,mo_i,mo_j,mo_k):
        all_tasks = [[ki,kj] for ki,kj in itertools.product(range(nkpts), repeat=2)]
        i0, i1, mo_e_oi, nonzero_vpaddingi = mo_i
        j0, j1, mo_e_oj, nonzero_vpaddingj = mo_j
        k0, k1, mo_e_ok, nonzero_vpaddingk = mo_k
        def _get_eijk(ki,kj):
            kk = kpts_helper.get_kconserv3(mycc._scf.cell, kpts, [ka,kb,kc,ki,kj])
            eijk = _get_epqr([i0,i1,ki,mo_e_oi,nonzero_vpaddingi],
                             [j0,j1,kj,mo_e_oj,nonzero_vpaddingj],
                             [k0,k1,kk,mo_e_ok,nonzero_vpaddingk],
                             fac=[1.,1.,1.])
            ind = (ki*nkpts+kj)*eijk.size+np.arange(eijk.size)
            return ind, eijk.ravel()
        shape = (nkpts,nkpts,i1-i0,j1-j0,k1-k0)
        eijk = frombatchfunc(_get_eijk, shape, all_tasks).array
        return eijk

    t2taa = t2aa.transpose(2,3,0,1)
    aimj = eris.ooov.conj().transpose(3,2,1,0)
    bcjk = eris.ovov.conj().transpose(1,3,0,2)

    energy_t = 0.0
    blkmin = 4
    vir_blksize = min(min(nvira,nvirb), max(blkmin, int((slice_size/nkpts**2)**(1./3)/max(nocca,noccb))))
    log.info("nvir = (%i, %i), virtual slice size: %i", nvira, nvirb, vir_blksize)
    tasks = []
    for a0, a1 in lib.prange(0, nvira, vir_blksize):
        for b0, b1 in lib.prange(0, nvira, vir_blksize):
            for c0, c1 in lib.prange(0, nvira, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka,kb,kc in itertools.product(range(nkpts), repeat=3):
        for orbslice in tasks:
            a0, a1, b0, b1, c0, c1 = orbslice
            w = get_w_s6(ka,kb,kc,orbslice,t2taa,eris.vovv,aimj)
            r = r6(w)
            w+= get_w_s6(kc,ka,kb,[c0,c1,a0,a1,b0,b1],t2taa,eris.vovv,aimj).transpose(1,2,0,4,5,3)
            w+= get_w_s6(kb,kc,ka,[b0,b1,c0,c1,a0,a1],t2taa,eris.vovv,aimj).transpose(2,0,1,5,3,4)
            w+= get_w_s6(ka,kc,kb,[a0,a1,c0,c1,b0,b1],t2taa,eris.vovv,aimj).transpose(0,2,1,3,5,4)
            w+= get_w_s6(kc,kb,ka,[c0,c1,b0,b1,a0,a1],t2taa,eris.vovv,aimj).transpose(2,1,0,5,4,3)
            w+= get_w_s6(kb,ka,kc,[b0,b1,a0,a1,c0,c1],t2taa,eris.vovv,aimj).transpose(1,0,2,4,3,5)
            w+= get_v_s6(ka,kb,kc,orbslice,t1a,t2taa,bcjk,fvo)
            w+= get_v_s6(kc,ka,kb,[c0,c1,a0,a1,b0,b1],t1a,t2taa,bcjk,fvo).transpose(1,2,0,4,5,3)
            w+= get_v_s6(kb,kc,ka,[b0,b1,c0,c1,a0,a1],t1a,t2taa,bcjk,fvo).transpose(2,0,1,5,3,4)
            w+= get_v_s6(ka,kc,kb,[a0,a1,c0,c1,b0,b1],t1a,t2taa,bcjk,fvo).transpose(0,2,1,3,5,4)
            w+= get_v_s6(kc,kb,ka,[c0,c1,b0,b1,a0,a1],t1a,t2taa,bcjk,fvo).transpose(2,1,0,5,4,3)
            w+= get_v_s6(kb,ka,kc,[b0,b1,a0,a1,c0,c1],t1a,t2taa,bcjk,fvo).transpose(1,0,2,4,3,5)
            eabc = _get_epqr([a0,a1,ka,mo_ea_v,nonzero_vpadding_a],
                             [b0,b1,kb,mo_ea_v,nonzero_vpadding_a],
                             [c0,c1,kc,mo_ea_v,nonzero_vpadding_a],
                             fac=[-1.,-1.,-1.])
            eabc = asarray(eabc)
            eijk = get_eijk(ka,kb,kc,mo_oa_script,mo_oa_script,mo_oa_script)
            d3 = eijk.reshape(nkpts,nkpts,1,1,1,nocca,nocca,nocca)+eabc.reshape(1,1,a1-a0,b1-b0,c1-c0,1,1,1)
            w = w/d3
            energy_t += einsum('ijkabc,ijkabc', w.conj(), r)

    cpu1 = log.timer('UCCSD(T) alpha-alpha-alpha block', *cpu1)

    t2tbb = t2bb.transpose(2,3,0,1)
    AIMJ = eris.OOOV.conj().transpose(3,2,1,0)
    BCJK = eris.OVOV.conj().transpose(1,3,0,2)
    tasks = []
    for a0, a1 in lib.prange(0, nvirb, vir_blksize):
        for b0, b1 in lib.prange(0, nvirb, vir_blksize):
            for c0, c1 in lib.prange(0, nvirb, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka,kb,kc in itertools.product(range(nkpts), repeat=3):
        for orbslice in tasks:
            a0, a1, b0, b1, c0, c1 = orbslice
            w = get_w_s6(ka,kb,kc,orbslice,t2tbb,eris.VOVV,AIMJ)
            r = r6(w)
            w+= get_w_s6(kc,ka,kb,[c0,c1,a0,a1,b0,b1],t2tbb,eris.VOVV,AIMJ).transpose(1,2,0,4,5,3)
            w+= get_w_s6(kb,kc,ka,[b0,b1,c0,c1,a0,a1],t2tbb,eris.VOVV,AIMJ).transpose(2,0,1,5,3,4)
            w+= get_w_s6(ka,kc,kb,[a0,a1,c0,c1,b0,b1],t2tbb,eris.VOVV,AIMJ).transpose(0,2,1,3,5,4)
            w+= get_w_s6(kc,kb,ka,[c0,c1,b0,b1,a0,a1],t2tbb,eris.VOVV,AIMJ).transpose(2,1,0,5,4,3)
            w+= get_w_s6(kb,ka,kc,[b0,b1,a0,a1,c0,c1],t2tbb,eris.VOVV,AIMJ).transpose(1,0,2,4,3,5)

            w+= get_v_s6(ka,kb,kc,orbslice,t1b,t2tbb,BCJK,fVO)
            w+= get_v_s6(kc,ka,kb,[c0,c1,a0,a1,b0,b1],t1b,t2tbb,BCJK,fVO).transpose(1,2,0,4,5,3)
            w+= get_v_s6(kb,kc,ka,[b0,b1,c0,c1,a0,a1],t1b,t2tbb,BCJK,fVO).transpose(2,0,1,5,3,4)
            w+= get_v_s6(ka,kc,kb,[a0,a1,c0,c1,b0,b1],t1b,t2tbb,BCJK,fVO).transpose(0,2,1,3,5,4)
            w+= get_v_s6(kc,kb,ka,[c0,c1,b0,b1,a0,a1],t1b,t2tbb,BCJK,fVO).transpose(2,1,0,5,4,3)
            w+= get_v_s6(kb,ka,kc,[b0,b1,a0,a1,c0,c1],t1b,t2tbb,BCJK,fVO).transpose(1,0,2,4,3,5)
            eabc = _get_epqr([a0,a1,ka,mo_eb_v,nonzero_vpadding_b],
                             [b0,b1,kb,mo_eb_v,nonzero_vpadding_b],
                             [c0,c1,kc,mo_eb_v,nonzero_vpadding_b],
                             fac=[-1.,-1.,-1.])
            eabc = asarray(eabc)
            eijk = get_eijk(ka,kb,kc,mo_ob_script,mo_ob_script,mo_ob_script)
            d3 = eijk.reshape(nkpts,nkpts,1,1,1,noccb,noccb,noccb)+eabc.reshape(1,1,a1-a0,b1-b0,c1-c0,1,1,1)
            w = w/d3
            energy_t += einsum('ijkabc,ijkabc', w.conj(), r)

    cpu1 = log.timer('UCCSD(T) beta-beta-beta block', *cpu1)

    t2pab = t2ab.transpose(0,1,3,2)
    t2tab = t2ab.transpose(2,3,0,1)
    t2xab = t2ab.transpose(3,2,1,0)
    AIck = eris.ovOV.conj().transpose(3,2,1,0)
    aiCK = eris.ovOV.conj().transpose(1,0,3,2)

    AImj = eris.ooOV.conj().transpose(3,2,1,0)
    aiMJ = eris.OOov.conj().transpose(3,2,1,0)

    tasks = []
    for a0, a1 in lib.prange(0, nvirb, vir_blksize):
        for b0, b1 in lib.prange(0, nvira, vir_blksize):
            for c0, c1 in lib.prange(0, nvira, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    def get_w_baa(ka,kb,kc,orbslice):
        a0,a1,b0,b1,c0,c1=orbslice
        ttmp = t2pab[:,:,ka,:,:,a0:a1]
        wtmp = eris.vovv[kc,:,kb,c0:c1,:,b0:b1]

        w = einsum('jiae,ckbe->abcijk',ttmp,wtmp)*2
        ttmp = t2ab[:,:,kb,:,:,b0:b1]
        wtmp = eris.voVV[kc,:,ka,c0:c1,:,a0:a1]

        w+= einsum('jibe,ckae->abcijk',ttmp,wtmp)*2
        ttmp = t2aa[:,:,kb,:,:,b0:b1]
        wtmp = eris.VOvv[ka,:,kc,a0:a1,:,c0:c1]

        w += einsum('jkbe,aice->abcijk', ttmp, wtmp)
        ttmp = t2tab[kb,ka,:,b0:b1,a0:a1]
        wtmp = aimj[kc,:,:,c0:c1]

        w -= einsum('bami,ckmj->abcijk', ttmp, wtmp) * 2
        wtmp = aiMJ[kc,:,:,c0:c1]

        w -= einsum('bajm,ckmi->abcijk', ttmp, wtmp) * 2
        ttmp = t2taa[kb,kc,:,b0:b1,c0:c1]
        wtmp = AImj[ka,:,:,a0:a1]
        w -= einsum('bcjm,aimk->abcijk', ttmp, wtmp)
        return w

    def get_v_baa(ka,kb,kc,orbslice):
        a0,a1,b0,b1,c0,c1=orbslice
        IA = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[ka]))
        JB = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[kb]))
        ttmp = einsum('Iia,I->Iia', t1b[:,:,a0:a1], IA)
        v = einsum('Jbcjk,Iia->IJabcijk', bcjk[kb,kc,:,b0:b1,c0:c1], ttmp)
        ttmp = einsum('Jjb,J->Jjb', t1a[:,:,b0:b1], JB)
        v+=einsum('Iaick,Jjb->IJabcijk', AIck[ka,:,kc,a0:a1,:,c0:c1], ttmp) *2
        ftmp = einsum('Iai,I->Iai', fVO[:,a0:a1], IA)
        v += einsum('Jbcjk,Iai->IJabcijk', t2taa[kb,kc,:,b0:b1,c0:c1], ftmp) * .5
        ftmp = einsum('Jbj,J->Jbj', fvo[:,b0:b1], JB)
        v += einsum('Iacik,Jbj->IJabcijk', t2xab[ka,kc,:,a0:a1,c0:c1], ftmp) * 2
        return array(v, mycc.gen_sym("000+++", kpts[ka]+kpts[kb]+kpts[kc]))

    for ka,kb,kc in itertools.product(range(nkpts), repeat=3):
        for orbslice in tasks:
            a0, a1, b0, b1, c0, c1 = orbslice
            w = get_w_baa(ka,kb,kc,orbslice)
            r = w - w.transpose(0,1,2,3,5,4)
            w0 = get_w_baa(ka,kc,kb,[a0,a1,c0,c1,b0,b1])
            r += w0.transpose(0,2,1,3,5,4)-  w0.transpose(0,2,1,3,4,5)
            w += get_v_baa(ka,kb,kc,orbslice)
            eabc = _get_epqr([a0,a1,ka,mo_eb_v,nonzero_vpadding_b],
                             [b0,b1,kb,mo_ea_v,nonzero_vpadding_a],
                             [c0,c1,kc,mo_ea_v,nonzero_vpadding_a],
                             fac=[-1.,-1.,-1.])
            eabc = asarray(eabc)
            eijk = get_eijk(ka,kb,kc,mo_ob_script,mo_oa_script,mo_oa_script)
            d3 = eijk.reshape(nkpts,nkpts,1,1,1,noccb,nocca,nocca)+eabc.reshape(1,1,a1-a0,b1-b0,c1-c0,1,1,1)
            r /= d3
            energy_t += einsum('abcijk,abcijk', w.conj(), r)

    cpu1 = log.timer('UCCSD(T) beta-alpha-alpha block', *cpu1)

    def get_w_abb(ka,kb,kc,orbslice):
        a0,a1,b0,b1,c0,c1=orbslice
        ttmp = t2ab[:,:,ka,:,:,a0:a1]
        wtmp = eris.VOVV[kc,:,kb,c0:c1,:,b0:b1]
        w = einsum('ijae,ckbe->abcijk', ttmp, wtmp) * 2

        ttmp = t2xab[kb,:,:,b0:b1]
        wtmp = eris.VOvv[kc,:,ka,c0:c1,:,a0:a1]
        w += einsum('beji,ckae->abcijk', ttmp, wtmp) * 2

        ttmp = t2bb[:,:,kb,:,:,b0:b1]
        wtmp = eris.voVV[ka,:,kc,a0:a1,:,c0:c1]
        w += einsum('jkbe,aice->abcijk', ttmp, wtmp)

        ttmp = t2tab[ka,kb,:,a0:a1,b0:b1]
        wtmp = AIMJ[kc,:,:,c0:c1]
        w -= einsum('abim,ckmj->abcijk', ttmp, wtmp) * 2

        wtmp = AImj[kc,:,:,c0:c1]
        w -= einsum('abmj,ckmi->abcijk', ttmp, wtmp) * 2

        ttmp = t2tbb[kb,kc,:,b0:b1,c0:c1]
        wtmp = aiMJ[ka,:,:,a0:a1]
        w -= einsum('bcjm,aimk->abcijk', ttmp, wtmp)
        return w

    def get_v_abb(ka,kb,kc,orbslice):
        a0,a1,b0,b1,c0,c1=orbslice
        IA = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[ka]))
        JB = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[kb]))
        ttmp = einsum('Iia,I->Iia', t1a[:,:,a0:a1], IA)
        v = einsum('Jbcjk,Iia->IJabcijk', BCJK[kb,kc,:,b0:b1,c0:c1], ttmp)
        ttmp = einsum('Jjb,J->Jjb', t1b[:,:,b0:b1], JB)
        v+=einsum('Iaick,Jjb->IJabcijk', aiCK[ka,:,kc,a0:a1,:,c0:c1], ttmp) *2
        ftmp = einsum('Iai,I->Iai', fvo[:,a0:a1], IA)
        v += einsum('Jbcjk,Iai->IJabcijk', t2tbb[kb,kc,:,b0:b1,c0:c1], ftmp) * .5
        ftmp = einsum('Jbj,J->Jbj', fVO[:,b0:b1], JB)
        v += einsum('Iacik,Jbj->IJabcijk', t2tab[ka,kc,:,a0:a1,c0:c1], ftmp) * 2
        return array(v, mycc.gen_sym("000+++", kpts[ka]+kpts[kb]+kpts[kc]))

    tasks = []
    for a0, a1 in lib.prange(0, nvira, vir_blksize):
        for b0, b1 in lib.prange(0, nvirb, vir_blksize):
            for c0, c1 in lib.prange(0, nvirb, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka,kb,kc in itertools.product(range(nkpts), repeat=3):
        for orbslice in tasks:
            a0, a1, b0, b1, c0, c1 = orbslice
            w = get_w_abb(ka,kb,kc,orbslice)
            r = w - w.transpose(0,1,2,3,5,4)
            w0 = get_w_abb(ka,kc,kb,[a0,a1,c0,c1,b0,b1])
            r += w0.transpose(0,2,1,3,5,4)-  w0.transpose(0,2,1,3,4,5)
            w += get_v_abb(ka,kb,kc,orbslice)
            eabc = _get_epqr([a0,a1,ka,mo_ea_v,nonzero_vpadding_a],
                             [b0,b1,kb,mo_eb_v,nonzero_vpadding_b],
                             [c0,c1,kc,mo_eb_v,nonzero_vpadding_b],
                             fac=[-1.,-1.,-1.])
            eabc = asarray(eabc)
            eijk = get_eijk(ka,kb,kc,mo_oa_script,mo_ob_script,mo_ob_script)
            d3 = eijk.reshape(nkpts,nkpts,1,1,1,nocca,noccb,noccb)+eabc.reshape(1,1,a1-a0,b1-b0,c1-c0,1,1,1)
            r /= d3
            energy_t += einsum('abcijk,abcijk', w.conj(), r)

    cpu1 = log.timer('UCCSD(T) alpha-beta-beta block', *cpu1)

    energy_t = energy_t /nkpts *.25

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of UCCSD(T) energy was found %s', energy_t.imag)
    log.timer('UCCSD(T)', *cpu0)
    log.note('UCCSD(T) correction per cell = %.15g', energy_t.real)
    log.note('UCCSD(T) correction per cell (imag) = %.15g', energy_t.imag)
    return energy_t.real

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
    cell = gto.Cell()
    cell.atom='''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.mesh = [13]*3
    cell.verbose= 5
    cell.build()

    kpts = cell.make_kpts([1,1,3]) + 0.01
    kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None)
    kmf.kernel()

    from pyscf.ctfcc.kccsd_uhf import KUCCSD
    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    et2 = kernel(mycc, eris)
    print(abs(et2--5.185576747969966e-06))
