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

'''
UCCSD with CTF as backend, see pyscf.cc.uccsd_slow
'''

import numpy as np
import time
from pyscf.lib import logger
from pyscf.cc import uccsd_slow
from pyscf import lib

from pyscf.ctfcc import rccsd
from pyscf.ctfcc.integrals.ao2mo import make_ao_ints
from pyscf.ctfcc import ctf_helper
from symtensor.ctf import einsum, frombatchfunc
from symtensor.ctf.backend import hstack, asarray, norm, diag, argsort

uccsd_slow.imd.einsum = uccsd_slow.einsum = einsum
uccsd_slow.asarray = asarray

def get_normt_diff(cc, t1, t2, t1new, t2new):
    normt = 0.0
    for old, new in zip(t1+t2, t1new+t2new):
        normt += norm(new-old)
    return normt

def amplitudes_to_vector(t1, t2):
    tavec = ctf_helper.amplitudes_to_vector_s4(t1[0], t2[0])
    tbvec = ctf_helper.amplitudes_to_vector_s4(t1[1], t2[2])
    vector = hstack((tavec, tbvec, t2[1].ravel()))
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    sizea = nocca * nvira + nocca*(nocca-1)//2*nvira*(nvira-1)//2
    sizeb = noccb * nvirb + noccb*(noccb-1)//2*nvirb*(nvirb-1)//2
    t1a, t2aa = ctf_helper.vector_to_amplitudes_s4(vector[:sizea], nmoa, nocca)
    t1b, t2bb = ctf_helper.vector_to_amplitudes_s4(vector[sizea:sizea+sizeb], nmob, noccb)
    t2ab = vector[sizea+sizeb:].reshape(nocca,noccb,nvira,nvirb)
    return (t1a,t1b), (t2aa,t2ab,t2bb)

def amplitudes_to_vector_ip(r1, r2):
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    r2avec = ctf_helper.pack_ip_r2(r2aaa)
    r2bvec = ctf_helper.pack_ip_r2(r2bbb)
    return hstack((r1a, r1b,
                   r2avec, r2baa.ravel(),
                   r2abb.ravel(), r2bvec))

def vector_to_amplitudes_ip(vector, nmo, nocc):
    '''For spin orbitals'''
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    sizes = (nocca, noccb, nocca*(nocca-1)//2*nvira, noccb*nocca*nvira,
             nocca*noccb*nvirb, noccb*(noccb-1)//2*nvirb)
    sections = np.cumsum(sizes[:-1])
    r1a = vector[:sections[0]]
    r1b = vector[sections[0]:sections[1]]
    r2a = vector[sections[1]:sections[2]]
    r2baa = vector[sections[2]:sections[3]].reshape(noccb,nocca,nvira)
    r2abb = vector[sections[3]:sections[4]].reshape(nocca,noccb,nvirb)
    r2b = vector[sections[4]:]
    r2aaa = ctf_helper.unpack_ip_r2(r2a, nmoa, nocca)
    r2bbb = ctf_helper.unpack_ip_r2(r2b, nmob, noccb)
    return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nocca, noccb = nocc
    nmoa, nmob = nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    sizes = (nvira, nvirb, nocca*nvira*(nvira-1)//2, nocca*nvirb*nvira,
             noccb*nvira*nvirb, noccb*nvirb*(nvirb-1)//2)
    sections = np.cumsum(sizes[:-1])
    r1a = vector[:sections[0]]
    r1b = vector[sections[0]:sections[1]]
    r2a = vector[sections[1]:sections[2]]
    r2aba = vector[sections[2]:sections[3]].reshape(nocca,nvirb,nvira)
    r2bab = vector[sections[3]:sections[4]].reshape(noccb,nvira,nvirb)
    r2b = vector[sections[4]:]
    r2aaa = ctf_helper.unpack_ea_r2(r2a, nmoa, nocca)
    r2bbb = ctf_helper.unpack_ea_r2(r2b, nmob, noccb)
    return (r1a, r1b), (r2aaa, r2aba, r2bab, r2bbb)

def amplitudes_to_vector_ea(r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    r2a = ctf_helper.pack_ea_r2(r2aaa)
    r2b = ctf_helper.pack_ea_r2(r2bbb)
    return hstack((r1a, r1b,\
                   r2a, r2aba.ravel(), \
                   r2bab.ravel(), r2b))

def uccsd_t(mycc, t1=None, t2=None, eris=None, slice_size=None, free_vvvv=False):
    '''
    Args:
        slice_size:
            the amount of memory(MB) to be used for slicing t3 amplitudes
            by default, it's set to the same size as vvvv integrals
        free_vvvv:
            a boolean for whether to free up the vvvv integrals in eris object,
            as vvvv is not needed in CCSD(T), default set to False
    '''
    cpu1 = cpu0 = (time.clock(), time.time())
    if t1 is None or t2 is None:
        t1, t2 = mycc.t1, mycc.t2

    log = logger.Logger(mycc.stdout, mycc.verbose)

    if eris is None:
        eris = mycc.ao2mo()

    if free_vvvv:
        eris.vvvv = eris.vvVV = eris.VVVV = None

    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) + w.transpose(1,2,0,3,4,5) \
                - w.transpose(2,1,0,3,4,5) - w.transpose(0,2,1,3,4,5) \
                - w.transpose(1,0,2,3,4,5))

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape

    eia = eris.eia
    eIA = eris.eIA
    fvo = eris.fov.transpose(1,0).conj()
    fVO = eris.fOV.transpose(1,0).conj()

    if slice_size is None:
        slice_size = max(nvira,nvirb)**4
    else:
        slice_size = int(slice_size /4. * 1.25e5)

    energy_t = 0.

    def get_w_s6(orbslice, it2, ivovv, iooov):
        a0,a1,b0,b1,c0,c1 = orbslice
        w = einsum('ijae,ckbe->ijkabc', it2[:,:,a0:a1], ivovv[c0:c1,:,b0:b1])
        w-= einsum('mkbc,jmia->ijkabc', it2[:,:,b0:b1,c0:c1], iooov[:,:,:,a0:a1])
        return w

    def get_v_s6(orbslice, it1, it2, iovov, ivo):
        a0,a1,b0,b1,c0,c1 = orbslice
        v = einsum('jbkc,ia->ijkabc', iovov[:,b0:b1,:,c0:c1], it1[:,a0:a1])
        v+= einsum('jkbc,ai->ijkabc', it2[:,:,b0:b1,c0:c1], ivo[a0:a1]) * .5
        return v

    ooov = eris.ooov.conj()
    ovov = eris.ovov.conj()

    blkmin = 4
    vir_blksize = min(min(nvira,nvirb), max(blkmin, int((slice_size)**(1./3)/max(nocca,noccb))))
    log.info("nvir=(%i, %i), virtual blksize %i", nvira, nvirb, vir_blksize)
    tasks = []
    for a0, a1 in lib.prange(0, nvira, vir_blksize):
        for b0, b1 in lib.prange(0, nvira, vir_blksize):
            for c0, c1 in lib.prange(0, nvira, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for orbslice in tasks:
        a0,a1,b0,b1,c0,c1 = orbslice
        d3 = eia[:,a0:a1].reshape(nocca,1,1,a1-a0,1,1) + \
             eia[:,b0:b1].reshape(1,nocca,1,1,b1-b0,1) + \
             eia[:,c0:c1].reshape(1,1,nocca,1,1,c1-c0)
        w = get_w_s6(orbslice, t2aa, eris.vovv, ooov)
        r = r6(w)
        w+= get_w_s6([c0,c1,a0,a1,b0,b1],t2aa,eris.vovv,ooov).transpose(1,2,0,4,5,3)
        w+= get_w_s6([b0,b1,c0,c1,a0,a1],t2aa,eris.vovv,ooov).transpose(2,0,1,5,3,4)
        w+= get_w_s6([a0,a1,c0,c1,b0,b1],t2aa,eris.vovv,ooov).transpose(0,2,1,3,5,4)
        w+= get_w_s6([c0,c1,b0,b1,a0,a1],t2aa,eris.vovv,ooov).transpose(2,1,0,5,4,3)
        w+= get_w_s6([b0,b1,a0,a1,c0,c1],t2aa,eris.vovv,ooov).transpose(1,0,2,4,3,5)
        w+= get_v_s6(orbslice,t1a,t2aa,ovov,fvo)
        w+= get_v_s6([c0,c1,a0,a1,b0,b1],t1a,t2aa,ovov,fvo).transpose(1,2,0,4,5,3)
        w+= get_v_s6([b0,b1,c0,c1,a0,a1],t1a,t2aa,ovov,fvo).transpose(2,0,1,5,3,4)
        w+= get_v_s6([a0,a1,c0,c1,b0,b1],t1a,t2aa,ovov,fvo).transpose(0,2,1,3,5,4)
        w+= get_v_s6([c0,c1,b0,b1,a0,a1],t1a,t2aa,ovov,fvo).transpose(2,1,0,5,4,3)
        w+= get_v_s6([b0,b1,a0,a1,c0,c1],t1a,t2aa,ovov,fvo).transpose(1,0,2,4,3,5)

        w = w/d3
        energy_t += einsum('ijkabc,ijkabc', w.conj(), r)

    OOOV = eris.OOOV.conj()
    OVOV = eris.OVOV.conj()

    tasks = []
    for a0, a1 in lib.prange(0, nvirb, vir_blksize):
        for b0, b1 in lib.prange(0, nvirb, vir_blksize):
            for c0, c1 in lib.prange(0, nvirb, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for orbslice in tasks:
        a0,a1,b0,b1,c0,c1 = orbslice
        d3 = eIA[:,a0:a1].reshape(noccb,1,1,a1-a0,1,1) + \
             eIA[:,b0:b1].reshape(1,noccb,1,1,b1-b0,1) + \
             eIA[:,c0:c1].reshape(1,1,noccb,1,1,c1-c0)
        w = get_w_s6(orbslice, t2bb, eris.VOVV, OOOV)
        r = r6(w)
        w+= get_w_s6([c0,c1,a0,a1,b0,b1],t2bb,eris.VOVV,OOOV).transpose(1,2,0,4,5,3)
        w+= get_w_s6([b0,b1,c0,c1,a0,a1],t2bb,eris.VOVV,OOOV).transpose(2,0,1,5,3,4)
        w+= get_w_s6([a0,a1,c0,c1,b0,b1],t2bb,eris.VOVV,OOOV).transpose(0,2,1,3,5,4)
        w+= get_w_s6([c0,c1,b0,b1,a0,a1],t2bb,eris.VOVV,OOOV).transpose(2,1,0,5,4,3)
        w+= get_w_s6([b0,b1,a0,a1,c0,c1],t2bb,eris.VOVV,OOOV).transpose(1,0,2,4,3,5)
        w+= get_v_s6(orbslice,t1b,t2bb,OVOV,fVO)
        w+= get_v_s6([c0,c1,a0,a1,b0,b1],t1b,t2bb,OVOV,fVO).transpose(1,2,0,4,5,3)
        w+= get_v_s6([b0,b1,c0,c1,a0,a1],t1b,t2bb,OVOV,fVO).transpose(2,0,1,5,3,4)
        w+= get_v_s6([a0,a1,c0,c1,b0,b1],t1b,t2bb,OVOV,fVO).transpose(0,2,1,3,5,4)
        w+= get_v_s6([c0,c1,b0,b1,a0,a1],t1b,t2bb,OVOV,fVO).transpose(2,1,0,5,4,3)
        w+= get_v_s6([b0,b1,a0,a1,c0,c1],t1b,t2bb,OVOV,fVO).transpose(1,0,2,4,3,5)

        w = w/d3
        energy_t += einsum('ijkabc,ijkabc', w.conj(), r)


    ovOV = eris.ovOV.conj()
    ooOV = eris.ooOV.conj()
    OOov = eris.OOov.conj()

    tasks = []
    for a0, a1 in lib.prange(0, nvirb, vir_blksize):
        for b0, b1 in lib.prange(0, nvira, vir_blksize):
            for c0, c1 in lib.prange(0, nvira, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    def get_w_baa(orbslice):
        a0,a1,b0,b1,c0,c1 = orbslice
        w  = einsum('jiea,ckbe->ijkabc', t2ab[:,:,:,a0:a1], eris.vovv[c0:c1,:,b0:b1]) * 2
        w += einsum('jibe,ckae->ijkabc', t2ab[:,:,b0:b1], eris.voVV[c0:c1,:,a0:a1]) * 2
        w += einsum('jkbe,aice->ijkabc', t2aa[:,:,b0:b1], eris.VOvv[a0:a1,:,c0:c1])
        w -= einsum('miba,jmkc->ijkabc', t2ab[:,:,b0:b1,a0:a1], ooov[:,:,:,c0:c1]) * 2
        w -= einsum('jmba,imkc->ijkabc', t2ab[:,:,b0:b1,a0:a1], OOov[:,:,:,c0:c1]) * 2
        w -= einsum('jmbc,kmia->ijkabc', t2aa[:,:,b0:b1,c0:c1], ooOV[:,:,:,a0:a1])
        return w


    for orbslice in tasks:
        a0,a1,b0,b1,c0,c1 = orbslice
        d3 = eIA[:,a0:a1].reshape(noccb,1,1,a1-a0,1,1) + \
             eia[:,b0:b1].reshape(1,nocca,1,1,b1-b0,1) + \
             eia[:,c0:c1].reshape(1,1,nocca,1,1,c1-c0)
        w = get_w_baa(orbslice)
        r = w - w.transpose(0,2,1,3,4,5)
        w0 = get_w_baa([a0,a1,c0,c1,b0,b1])
        r += w0.transpose(0,2,1,3,5,4)-  w0.transpose(0,1,2,3,5,4)

        w += einsum('jbkc,ia->ijkabc', ovov[:,b0:b1,:,c0:c1], t1b[:,a0:a1])
        w += einsum('kcia,jb->ijkabc', ovOV[:,c0:c1,:,a0:a1], t1a[:,b0:b1]) * 2
        w += einsum('jkbc,ai->ijkabc', t2aa[:,:,b0:b1,c0:c1], fVO[a0:a1]) * .5
        w += einsum('kica,bj->ijkabc', t2ab[:,:,c0:c1,a0:a1], fvo[b0:b1]) * 2

        r /= d3
        energy_t += einsum('ijkabc,ijkabc', w.conj(), r)

    tasks = []
    for a0, a1 in lib.prange(0, nvira, vir_blksize):
        for b0, b1 in lib.prange(0, nvirb, vir_blksize):
            for c0, c1 in lib.prange(0, nvirb, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    def get_w_abb(orbslice):
        a0,a1,b0,b1,c0,c1 = orbslice
        w  = einsum('ijae,ckbe->ijkabc', t2ab[:,:,a0:a1], eris.VOVV[c0:c1,:,b0:b1]) * 2
        w += einsum('ijeb,ckae->ijkabc', t2ab[:,:,:,b0:b1], eris.VOvv[c0:c1,:,a0:a1]) * 2
        w += einsum('jkbe,aice->ijkabc', t2bb[:,:,b0:b1], eris.voVV[a0:a1,:,c0:c1])
        w -= einsum('imab,jmkc->ijkabc', t2ab[:,:,a0:a1,b0:b1], OOOV[:,:,:,c0:c1]) * 2
        w -= einsum('mjab,imkc->ijkabc', t2ab[:,:,a0:a1,b0:b1], ooOV[:,:,:,c0:c1]) * 2
        w -= einsum('jmbc,kmia->ijkabc', t2bb[:,:,b0:b1,c0:c1], OOov[:,:,:,a0:a1])
        return w

    for orbslice in tasks:
        a0,a1,b0,b1,c0,c1 = orbslice
        d3 = eia[:,a0:a1].reshape(nocca,1,1,a1-a0,1,1) + \
             eIA[:,b0:b1].reshape(1,noccb,1,1,b1-b0,1) + \
             eIA[:,c0:c1].reshape(1,1,noccb,1,1,c1-c0)
        w = get_w_abb(orbslice)
        r = w - w.transpose(0,2,1,3,4,5)
        w0 = get_w_abb([a0,a1,c0,c1,b0,b1])
        r += w0.transpose(0,2,1,3,5,4)-  w0.transpose(0,1,2,3,5,4)
        w += einsum('jbkc,ia->ijkabc', OVOV[:,b0:b1,:,c0:c1], t1a[:,a0:a1])
        w += einsum('iakc,jb->ijkabc', ovOV[:,a0:a1,:,c0:c1], t1b[:,b0:b1]) *2
        w += einsum('jkbc,ai->ijkabc', t2bb[:,:,b0:b1,c0:c1], fvo[a0:a1]) * .5
        w += einsum('ikac,bj->ijkabc', t2ab[:,:,a0:a1,c0:c1], fVO[b0:b1]) * 2
        r /= d3
        energy_t += einsum('ijkabc,ijkabc', w.conj(), r)

    energy_t *= .25
    log.timer('UCCSD(T)', *cpu0)
    log.note('UCCSD(T) correction = %.15g', energy_t.real)
    if abs(energy_t.imag) > 1e-4:
        log.note('UCCSD(T) correction (imag) = %.15g', energy_t.imag)
    return energy_t

class UCCSD(uccsd_slow.UCCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ctf_helper.synchronize(mf, ['mo_coeff', 'mo_energy', 'mo_occ'])
        uccsd_slow.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    get_normt_diff = get_normt_diff

    ccsd = rccsd.RCCSD.ccsd
    ipccsd = rccsd.RCCSD.ipccsd
    eaccsd = rccsd.RCCSD.eaccsd
    ccsd_t = uccsd_t

    def get_init_guess_ip(self, nroots=1, koopmans=False, diag=None):
        if diag is None: diag = self.ipccsd_diag()
        size = self.nip()
        if koopmans:
            nocca, noccb = self.nocc
            idx = argsort(diag[:nocca+noccb])[:nroots]
        else:
            idx = argsort(diag)[:nroots]
        def write_guess(i):
            return i*size+idx[i], np.ones(1)
        all_tasks = np.arange(nroots)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, all_tasks).array
        return guess

    def get_init_guess_ea(self, nroots=1, koopmans=False, diag=None):
        if diag is None: diag = self.eaccsd_diag()
        size = self.nea()
        if koopmans:
            nocca, noccb = self.nocc
            nmoa, nmob = self.nmo
            nvira, nvirb = nmoa-nocca, nmob-noccb
            idx = argsort(diag[:nvira+nvirb])[:nroots]
        else:
            idx = argsort(diag)[:nroots]
        def write_guess(i):
            return i*size+idx[i], np.ones(1)
        all_tasks = np.arange(nroots)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, all_tasks).array
        return guess

    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                rccsd.lambda_kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose,
                                    fintermediates=uccsd_slow.make_intermediates,
                                    fupdate=uccsd_slow.update_lambda)
        return self.l1, self.l2

    def amplitudes_to_vector_ip(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_to_amplitudes_ip(self, vector, **kwargs):
        nmo = self.nmo
        nocc = self.nocc
        return vector_to_amplitudes_ip(vector, nmo, nocc)

    def amplitudes_to_vector_ea(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_to_amplitudes_ea(self, vector, **kwargs):
        nmo = self.nmo
        nocc = self.nocc
        return vector_to_amplitudes_ea(vector, nmo, nocc)

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = uccsd_slow._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb
    eris.focka = asarray(eris.focka)
    eris.fockb = asarray(eris.fockb)

    eris.foo = eris.focka[:nocca,:nocca]
    eris.fov = eris.focka[:nocca,nocca:]
    eris.fvv = eris.focka[nocca:,nocca:]

    eris.fOO = eris.fockb[:noccb,:noccb]
    eris.fOV = eris.fockb[:noccb,noccb:]
    eris.fVV = eris.fockb[noccb:,noccb:]

    eris._foo = diag(diag(eris.foo))
    eris._fOO = diag(diag(eris.fOO))
    eris._fvv = diag(diag(eris.fvv))
    eris._fVV = diag(diag(eris.fVV))

    mo_ea, mo_eb = eris.mo_energy[0].real, eris.mo_energy[1].real
    eris.eia = mo_ea[:nocca][:,None] - mo_ea[nocca:][None,:]
    eris.eIA = mo_eb[:noccb][:,None] - mo_eb[noccb:][None,:]
    eris.eia = asarray(eris.eia)
    eris.eIA = asarray(eris.eIA)
    eris.eijab = eris.eia.reshape(nocca,1,nvira,1) + eris.eia.reshape(1,nocca,1,nvira)
    eris.eiJaB = eris.eia.reshape(nocca,1,nvira,1) + eris.eIA.reshape(1,noccb,1,nvirb)
    eris.eIJAB = eris.eIA.reshape(noccb,1,nvirb,1) + eris.eIA.reshape(1,noccb,1,nvirb)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    cput1 = (time.clock(), time.time())
    ppoo, ppov, ppvv = make_ao_ints(eris.mol, moa, nocca)
    cput1 = logger.timer(mycc, 'making ao integrals for alpha', *cput1)


    moa = asarray(moa)
    orba_o, orba_v = moa[:,:nocca], moa[:,nocca:]


    tmp = einsum('uvmn,ui->ivmn', ppoo, orba_o.conj())
    eris.oooo = einsum('ivmn,vj->ijmn', tmp, orba_o)
    eris.ooov = einsum('ivmn,va->mnia', tmp, orba_v)

    tmp = einsum('uvma,vb->ubma', ppov, orba_v)
    eris.ovov = einsum('ubma,ui->ibma', tmp, orba_o.conj())
    tmp = einsum('uvma,ub->mabv', ppov, orba_v.conj())
    eris.voov = einsum('mabv,vi->bima', tmp, orba_o)

    tmp = einsum('uvab,ui->ivab', ppvv, orba_o.conj())
    eris.oovv = einsum('ivab,vj->ijab', tmp, orba_o)

    tmp = einsum('uvab,vc->ucab', ppvv, orba_v)
    eris.vovv = einsum('ucab,ui->ciba', tmp.conj(), orba_o)
    eris.vvvv = einsum('ucab,ud->dcab', tmp, orba_v.conj())

    del ppoo, ppov, ppvv

    cput1 = (time.clock(), time.time())
    ppOO, ppOV, ppVV = make_ao_ints(eris.mol, mob, noccb)
    cput1 = logger.timer(mycc, 'making ao integrals for beta', *cput1)
    mob = asarray(mob)
    orbb_o, orbb_v = mob[:,:noccb], mob[:,noccb:]
    tmp = einsum('uvmn,ui->ivmn', ppOO, orbb_o.conj())
    eris.OOOO = einsum('ivmn,vj->ijmn', tmp, orbb_o)
    eris.OOOV = einsum('ivmn,va->mnia', tmp, orbb_v)

    tmp = einsum('uvma,vb->ubma', ppOV, orbb_v)
    eris.OVOV = einsum('ubma,ui->ibma', tmp, orbb_o.conj())
    tmp = einsum('uvma,ub->mabv', ppOV, orbb_v.conj())
    eris.VOOV = einsum('mabv,vi->bima', tmp, orbb_o)

    tmp = einsum('uvab,ui->ivab', ppVV, orbb_o.conj())
    eris.OOVV = einsum('ivab,vj->ijab', tmp, orbb_o)

    tmp = einsum('uvab,vc->ucab', ppVV, orbb_v)
    eris.VOVV = einsum('ucab,ui->ciba', tmp.conj(), orbb_o)
    eris.VVVV = einsum('ucab,ud->dcab', tmp, orbb_v.conj())

    eris.ooOO = einsum('uvmn,ui,vj->ijmn', ppOO, orba_o.conj(), orba_o)
    eris.ooOV = einsum('uvma,ui,vj->ijma', ppOV, orba_o.conj(), orba_o)
    eris.ovOV = einsum('uvma,ui,vb->ibma', ppOV, orba_o.conj(), orba_v)
    eris.voOV = einsum('uvma,ub,vi->bima', ppOV, orba_v.conj(), orba_o)
    eris.ooVV = einsum('uvab,ui,vj->ijab', ppVV, orba_o.conj(), orba_o)
    eris.voVV = einsum('uvab,uc,vi->ciab', ppVV, orba_v.conj(), orba_o)
    eris.vvVV = einsum('uvab,uc,vd->cdab', ppVV, orba_v.conj(), orba_v)

    eris.OOoo = None
    eris.OOov = einsum('uvmn,ui,va->mnia', ppOO, orba_o.conj(), orba_v)
    eris.OOvv = einsum('uvmn,ua,vb->mnab', ppOO, orba_v.conj(), orba_v)
    eris.OVov = eris.ovOV.transpose(2,3,0,1)
    eris.VOov = eris.voOV.transpose(3,2,1,0).conj()
    eris.VOvv = einsum('uvma,ub,vc->amcb', ppOV, orba_v.conj(), orba_v).conj()
    del ppOO, ppOV, ppVV
    return eris

uccsd_slow._make_eris_incore = _make_eris_incore


if __name__ == '__main__':
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 5
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run(conv_tol=1e-14)

    mycc = UCCSD(mf)
    eris = mycc.ao2mo()
    e, t1, t2 = mycc.kernel(eris=eris)
    print(e - -0.2133432467414933)

    et1 = mycc.uccsd_t(t1, t2, eris)
    print(et1 - -0.0030600233005741453)

    et1 = mycc.ccsd_t_slow(t1, t2, eris)
    print(et1 - -0.0030600233005741453)

    eip, vip = mycc.ipccsd(nroots=8)
    print(eip[0] - 0.43356041)
    print(eip[2] - 0.51876597)

    eea, vea = mycc.eaccsd(nroots=8)
    print(eea[0] - 0.16737886)
    print(eea[2] - 0.24027623)
