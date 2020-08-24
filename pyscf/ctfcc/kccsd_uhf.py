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
KUCCSD with CTF as backend, all integrals in memory
'''

import numpy as np
import time
from functools import reduce
import itertools
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import uccsd_slow
import pyscf.pbc.tools.pbc as tools
from pyscf.pbc.cc import kccsd_uhf, eom_kccsd_uhf
from pyscf.pbc.mp.kump2 import (get_frozen_mask, get_nocc, get_nmo,
                                padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.cc.ccsd import _adjust_occ
from pyscf.pbc.cc.kccsd_rhf import _get_epq

from pyscf.ctfcc import uccsd, kccsd_rhf
from pyscf.ctfcc.integrals import ao2mo
from symtensor.ctf import einsum, array, frombatchfunc, zeros
from symtensor.ctf.backend import hstack, asarray, eye, argsort
from symtensor.symlib import SYMLIB

SLICE_SIZE = getattr(__config__, 'ctfcc_kccsd_slice_size', 4000)

from pyscf.ctfcc import ctf_helper
rank = ctf_helper.rank

def amplitudes_to_vector(t1, t2):
    return hstack((t1[0].ravel(), t1[1].ravel(),
                   t2[0].ravel(), t2[1].ravel(), t2[2].ravel()))

def energy(mycc, t1, t2, eris):
    return uccsd_slow.energy(mycc, t1, t2, eris, 1./mycc.nkpts)

def amplitudes_to_vector_ip(r1, r2):
    r1a, r1b = r1
    r2aaa, r2baa, r2abb, r2bbb = r2
    nkpts = r2aaa.array.shape[0]
    nocca, noccb = r1a.shape[0], r1b.shape[0]
    nvira, nvirb = r2aaa.shape[-1], r2bbb.shape[-1]
    r2aaa = r2aaa.array.transpose(0,2,1,3,4).reshape(nkpts*nocca,nkpts*nocca,nvira)
    r2bbb = r2bbb.array.transpose(0,2,1,3,4).reshape(nkpts*noccb,nkpts*noccb,nvirb)
    r2a = ctf_helper.pack_ip_r2(r2aaa)
    r2b = ctf_helper.pack_ip_r2(r2bbb)
    return hstack((r1a.ravel(), r1b.ravel(), r2a, r2baa.ravel(), r2abb.ravel(), r2b))

def amplitudes_to_vector_ea(r1, r2):
    r1a, r1b = r1
    r2aaa, r2aba, r2bab, r2bbb = r2
    nkpts = r2aaa.array.shape[0]
    nvira, nvirb = r1a.shape[0], r1b.shape[0]
    nocca, noccb = r2aaa.shape[0], r2bbb.shape[0]

    r2aaa = r2aaa.transpose(1,2,0).array.transpose(0,2,1,3,4).reshape(nkpts*nvira,nkpts*nvira,nocca)
    r2a = ctf_helper.pack_ea_r2(r2aaa.transpose(2,0,1))
    r2bbb = r2bbb.transpose(1,2,0).array.transpose(0,2,1,3,4).reshape(nkpts*nvirb,nkpts*nvirb,noccb)
    r2b = ctf_helper.pack_ea_r2(r2bbb.transpose(2,0,1))
    return hstack((r1a.ravel(), r1b.ravel(), r2a, r2aba.ravel(), r2bab.ravel(), r2b))

class KUCCSD(kccsd_rhf.KRCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, slice_size=SLICE_SIZE):
        ctf_helper.synchronize(mf, ["mo_coeff", "mo_occ", "mo_energy"])
        kccsd_uhf.KUCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.ip_partition = self.ea_partition = None
        self.slice_size = SLICE_SIZE
        self.max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)
        self.symlib = SYMLIB('ctf')
        self.__imds__ = None
        self._keys = self._keys.union(['max_space', 'ip_partition', '__imds__'\
                                       'ea_partition', 'symlib', 'slice_size'])
        self.make_symlib()

    energy = energy
    dump_flags = kccsd_uhf.KUCCSD.dump_flags
    nip = eom_kccsd_uhf.EOMIP.vector_size
    nea = eom_kccsd_uhf.EOMEA.vector_size
    update_amps = uccsd.UCCSD.update_amps
    ipccsd_matvec = uccsd.UCCSD.ipccsd_matvec
    eaccsd_matvec = uccsd.UCCSD.eaccsd_matvec
    solve_lambda = uccsd.UCCSD.solve_lambda

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb=  nmoa - nocca, nmob - noccb
        t1a = zeros([nocca,nvira], sym=self._sym[0])
        t1b = zeros([noccb,nvirb], sym=self._sym[0])

        t2aa = eris.ovov.conj().transpose(0,2,1,3) / eris.eijab
        t2aa-= eris.ovov.conj().transpose(2,0,1,3) / eris.eijab
        t2ab = eris.ovOV.conj().transpose(0,2,1,3) / eris.eiJaB
        t2bb = eris.OVOV.conj().transpose(0,2,1,3) / eris.eIJAB
        t2bb-= eris.OVOV.conj().transpose(2,0,1,3) / eris.eIJAB

        d = 0.0 + 0.j
        d += 0.25*(einsum('iajb,ijab->',eris.ovov,t2aa)
                 - einsum('jaib,ijab->',eris.ovov,t2aa))

        d += einsum('iajb,ijab->',eris.ovOV,t2ab)

        d += 0.25*(einsum('iajb,ijab->',eris.OVOV,t2bb)
                 - einsum('jaib,ijab->',eris.OVOV,t2bb))

        self.emp2 = d/self.nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, (t1a, t1b), (t2aa,t2ab,t2bb)

    def get_normt_diff(self, t1, t2, t1new, t2new):
        normt = 0.0
        for oldt, newt in zip(t1+t2, t1new+t2new):
            normt += (newt-oldt).norm()
        return normt

    def vector_to_amplitudes(self, vec, **kwargs):
        nkpts = self.nkpts
        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa - nocca, nmob - noccb
        sizes = (nkpts*nocca*nvira, nkpts*noccb*nvirb,
                 nkpts**3*nocca**2*nvira**2, nkpts**3*nocca*noccb*nvira*nvirb,
                 nkpts**3*noccb**2*nvirb**2)
        sections = np.cumsum(sizes[:-1])

        sym1, sym2 = self._sym[:2]
        t1a = array(vec[:sections[0]].reshape(nkpts,nocca,nvira), sym=sym1)
        t1b = array(vec[sections[0]:sections[1]].reshape(nkpts,noccb,nvirb), sym=sym1)

        t2aa = array(vec[sections[1]:sections[2]].reshape(nkpts,\
                            nkpts,nkpts,nocca,nocca,nvira,nvira), sym=sym2)
        t2ab = array(vec[sections[2]:sections[3]].reshape(nkpts,\
                            nkpts,nkpts,nocca,noccb,nvira,nvirb), sym=sym2)
        t2bb = array(vec[sections[3]:].reshape(nkpts,nkpts,nkpts,\
                            noccb,noccb,nvirb,nvirb), sym=sym2)

        t1a.symlib = t1b.symlib = t2aa.symlib =\
        t2ab.symlib = t2bb.symlib = self.symlib

        return (t1a,t1b), (t2aa,t2ab,t2bb)

    def amplitudes_to_vector(self, t1, t2, **kwargs):
        return amplitudes_to_vector(t1, t2)

    def ao2mo(self, mo_coeff=None):
        return _make_eris_incore(self, mo_coeff)

    def ccsd_t(self, t1=None, t2=None, eris=None, slice_size=None):
        if slice_size is None: slice_size = self.slice_size
        from pyscf.ctfcc import kccsd_t_uhf
        return kccsd_t_uhf.kernel(self, eris, t1, t2, slice_size)

    def ccsd_t_slow(self, t1=None, t2=None, eris=None, slice_size=None):
        raise NotImplementedError

    @property
    def imds(self):
        if self.__imds__ is None:
            self.__imds__ = uccsd_slow._IMDS(self)
        return self.__imds__

    def amplitudes_to_vector_ip(self, r1, r2, **kwargs):
        return amplitudes_to_vector_ip(r1, r2)

    def amplitudes_to_vector_ea(self, r1, r2, **kwargs):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_to_amplitudes_ip(self, vector, kshift=0):
        kpti = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpti)
        sym2 = self.gen_sym('++-', kpti)

        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nkpts = self.nkpts
        nvira, nvirb = nmoa-nocca, nmob-noccb
        sizes = (nocca, noccb, nkpts*nocca*(nkpts*nocca-1)//2*nvira,
                 nkpts**2*noccb*nocca*nvira, nkpts**2*nocca*noccb*nvirb, \
                 nkpts*noccb*(nkpts*noccb-1)//2*nvirb)
        sections = np.cumsum(sizes[:-1])
        r1a = array(vector[:sections[0]], sym1)
        r1b = array(vector[sections[0]:sections[1]], sym1)
        r2aaa = ctf_helper.unpack_ip_r2(vector[sections[1]:sections[2]], nkpts*nocca+nvira, nkpts*nocca).reshape(nkpts,nocca,nkpts,nocca,nvira).transpose(0,2,1,3,4)
        r2aaa = array(r2aaa, sym2)
        r2baa = array(vector[sections[2]:sections[3]].reshape(nkpts,nkpts,noccb,nocca,nvira), sym2)
        r2abb = array(vector[sections[3]:sections[4]].reshape(nkpts,nkpts,nocca,noccb,nvirb), sym2)
        r2bbb = ctf_helper.unpack_ip_r2(vector[sections[4]:], nkpts*noccb+nvirb, nkpts*noccb).reshape(nkpts,noccb,nkpts,noccb,nvirb).transpose(0,2,1,3,4)
        r2bbb = array(r2bbb, sym2)
        return (r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)

    def vector_to_amplitudes_ea(self, vector, kshift=0):
        kpta = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpta)
        sym2 = self.gen_sym('++-', kpta)
        sym3 = self.gen_sym('-++', kpta)

        nocca, noccb = self.nocc
        nmoa, nmob = self.nmo
        nkpts = self.nkpts
        nvira, nvirb = nmoa-nocca, nmob-noccb
        sizes = (nvira, nvirb, nkpts*nvira*(nkpts*nvira-1)//2*nocca, \
                 nkpts**2*nocca*nvirb*nvira, nkpts**2*noccb*nvira*nvirb, \
                 nkpts*nvirb*(nkpts*nvirb-1)//2*noccb)
        sections = np.cumsum(sizes[:-1])
        r1a = array(vector[:sections[0]], sym1)
        r1b = array(vector[sections[0]:sections[1]], sym1)

        r2aaa = ctf_helper.unpack_ea_r2(vector[sections[1]:sections[2]], nkpts*nvira+nocca, nocca).reshape(nocca,nkpts,nvira,nkpts,nvira).transpose(1,3,2,4,0)
        r2aaa = array(r2aaa, sym2).transpose(2,0,1)

        r2aba = array(vector[sections[2]:sections[3]].reshape(nkpts,nkpts,nocca,nvirb,nvira), sym3)
        r2bab = array(vector[sections[3]:sections[4]].reshape(nkpts,nkpts,noccb,nvira,nvirb), sym3)

        r2bbb = ctf_helper.unpack_ea_r2(vector[sections[4]:], nkpts*nvirb+noccb, noccb).reshape(noccb,nkpts,nvirb,nkpts,nvirb).transpose(1,3,2,4,0)
        r2bbb = array(r2bbb, sym2).transpose(2,0,1)

        return (r1a, r1b), (r2aaa, r2aba, r2bab, r2bbb)

    def ipccsd_diag(self, kshift):
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        t2aa, t2ab, t2bb = imds.t2
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('++-', self.kpts[kshift])

        Hr1a = array(-imds.Foo.diagonal()[kshift], sym1)
        Hr1b = array(-imds.FOO.diagonal()[kshift], sym1)
        IJA = self.symlib.get_irrep_map(sym2)
        nkpts = self.nkpts
        nocca, noccb, nvira, nvirb = t2ab.shape


        Fvv = imds.Fvv.diagonal()
        Foo = imds.Foo.diagonal()
        FVV = imds.FVV.diagonal()
        FOO = imds.FOO.diagonal()

        Hr2aaa = -Foo.reshape(nkpts,1,nocca,1,1) - Foo.reshape(1,nkpts,1,nocca,1) +\
                einsum('Aa,IJA->IJa', Fvv, IJA).reshape(nkpts,nkpts,1,1,nvira)

        Hr2baa = -FOO.reshape(nkpts,1,noccb,1,1) - Foo.reshape(1,nkpts,1,nocca,1) +\
                     einsum('Aa,IJA->IJa', Fvv, IJA).reshape(nkpts,nkpts,1,1,nvira)
        Hr2abb = -Foo.reshape(nkpts,1,nocca,1,1) - FOO.reshape(1,nkpts,1,noccb,1) +\
                     einsum('Aa,IJA->IJa', FVV, IJA).reshape(nkpts,nkpts,1,1,nvirb)
        Hr2bbb = -FOO.reshape(nkpts,1,noccb,1,1) - FOO.reshape(1,nkpts,1,noccb,1) +\
                     einsum('Aa,IJA->IJa', FVV, IJA).reshape(nkpts,nkpts,1,1,nvirb)
        if self.ip_partition != 'mp':
            Hr2aaa = Hr2aaa + einsum('IIJiijj->IJij',imds.Woooo).reshape(nkpts,nkpts,nocca,nocca,1)
            Hr2abb = Hr2abb + einsum('IIJiijj->IJij',imds.WooOO).reshape(nkpts,nkpts,nocca,noccb,1)
            Hr2bbb = Hr2bbb + einsum('IIJiijj->IJij',imds.WOOOO).reshape(nkpts,nkpts,noccb,noccb,1)
            Hr2baa = Hr2baa + einsum('JJIjjii->IJij',imds.WooOO).reshape(nkpts,nkpts,noccb,nocca,1)
            Hr2aaa -= einsum('IJiejb,IJijeb->IJijb', imds.Wovov[:,kshift], t2aa[:,:,kshift])
            Hr2abb -= einsum('IJiejb,IJijeb->IJijb', imds.WovOV[:,kshift], t2ab[:,:,kshift])
            Wtmp = imds.WovOV.transpose(2,3,0,1)
            Ttmp = t2ab.transpose(0,1,3,2)
            Hr2baa -= einsum('IJiejb,JIjieb->IJijb', Wtmp[:,kshift], Ttmp[:,:,kshift])
            Hr2bbb -= einsum('IJiejb,IJijeb->IJijb', imds.WOVOV[:,kshift], t2bb[:,:,kshift])
            Hr2aaa = Hr2aaa + einsum('IBBibbi,IJB->IJib', imds.Wovvo, IJA).reshape(nkpts,nkpts,nocca,1,nvira)
            Hr2aaa = Hr2aaa + einsum('JBBjbbj,IJB->IJjb', imds.Wovvo, IJA).reshape(nkpts,nkpts,1,nocca,nvira)
            Hr2baa = Hr2baa + einsum('JBBjbbj,IJB->IJjb', imds.Wovvo, IJA).reshape(nkpts,nkpts,1,nocca,nvira)
            Hr2baa = Hr2baa - einsum('IIBiibb,IJB->IJib', imds.WOOvv, IJA).reshape(nkpts,nkpts,noccb,1,nvira)
            Hr2abb = Hr2abb + einsum('JBBjbbj,IJB->IJjb', imds.WOVVO, IJA).reshape(nkpts,nkpts,1,noccb,nvirb)
            Hr2abb = Hr2abb - einsum('IIBiibb,IJB->IJib', imds.WooVV, IJA).reshape(nkpts,nkpts,nocca,1,nvirb)
            Hr2bbb = Hr2bbb + einsum('IBBibbi,IJB->IJib', imds.WOVVO, IJA).reshape(nkpts,nkpts,noccb,1,nvirb)
            Hr2bbb = Hr2bbb + einsum('JBBjbbj,IJB->IJjb', imds.WOVVO, IJA).reshape(nkpts,nkpts,1,noccb,nvirb)

        Hr2aaa = array(Hr2aaa, sym2)
        Hr2baa = array(Hr2baa, sym2)
        Hr2abb = array(Hr2abb, sym2)
        Hr2bbb = array(Hr2bbb, sym2)
        return self.amplitudes_to_vector_ip([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb])

    def eaccsd_diag(self, kshift):
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds
        t2aa, t2ab, t2bb = imds.t2
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('-++', self.kpts[kshift])

        Hr1a = array(imds.Fvv.diagonal()[kshift], sym1)
        Hr1b = array(imds.FVV.diagonal()[kshift], sym1)
        IAB = self.symlib.get_irrep_map(sym2)
        nkpts = self.nkpts
        nocca, noccb, nvira, nvirb = t2ab.shape

        Hr2aaa = (-imds.Foo.diagonal().reshape(nkpts,1,nocca,1,1) +\
                   imds.Fvv.diagonal().reshape(1,nkpts,1,nvira,1) +\
                   einsum('Bb,IAB->IAb', imds.Fvv.diagonal(), IAB).reshape(nkpts,nkpts,1,1,nvira))

        Hr2aba = (-imds.Foo.diagonal().reshape(nkpts,1,nocca,1,1) +\
                   imds.FVV.diagonal().reshape(1,nkpts,1,nvirb,1) +\
                   einsum('Bb,IAB->IAb', imds.Fvv.diagonal(), IAB).reshape(nkpts,nkpts,1,1,nvira))

        Hr2bab = (-imds.FOO.diagonal().reshape(nkpts,1,noccb,1,1) +\
                   imds.Fvv.diagonal().reshape(1,nkpts,1,nvira,1) +\
                   einsum('Bb,IAB->IAb', imds.FVV.diagonal(), IAB).reshape(nkpts,nkpts,1,1,nvirb))

        Hr2bbb = (-imds.FOO.diagonal().reshape(nkpts,1,noccb,1,1) +\
                   imds.FVV.diagonal().reshape(1,nkpts,1,nvirb,1) +\
                   einsum('Bb,IAB->IAb', imds.FVV.diagonal(), IAB).reshape(nkpts,nkpts,1,1,nvirb))

        if self.ea_partition != 'mp':
            Hr2aaa = Hr2aaa +einsum('AABaabb,IAB->IAab', imds.Wvvvv, IAB).reshape(nkpts,nkpts,1,nvira,nvira)
            Hr2aba = Hr2aba +einsum('BBAbbaa,IAB->IAab', imds.WvvVV, IAB).reshape(nkpts,nkpts,1,nvirb,nvira)
            Hr2bab = Hr2bab +einsum('AABaabb,IAB->IAab', imds.WvvVV, IAB).reshape(nkpts,nkpts,1,nvira,nvirb)
            Hr2bbb = Hr2bbb +einsum('AABaabb,IAB->IAab', imds.WVVVV, IAB).reshape(nkpts,nkpts,1,nvirb,nvirb)

            # Wovov term (physicist's Woovv)
            Hr2aaa -= einsum('AJkajb,JAkjab->JAjab', imds.Wovov[kshift], t2aa[kshift])
            Hr2aba -= einsum('JBjbka,JBjkba,JAB->JAjab', imds.WovOV[:,:,kshift], t2ab[:,kshift], IAB)
            Hr2bab -= einsum('AJkajb,JAkjab->JAjab', imds.WovOV[kshift], t2ab[kshift])
            Hr2bbb -= einsum('AJkajb,JAkjab->JAjab', imds.WOVOV[kshift], t2bb[kshift])

            # Wovvo term
            Hr2aaa = Hr2aaa + einsum('JBBjbbj,JAB->JAjb',imds.Wovvo, IAB).reshape(nkpts,nkpts,nocca,1,nvira)
            Hr2aaa = Hr2aaa + einsum('JAAjaaj->JAja', imds.Wovvo).reshape(nkpts,nkpts,nocca,nvira,1)

            Hr2aba = Hr2aba + einsum('JBBjbbj,JAB->JAjb', imds.Wovvo, IAB).reshape(nkpts,nkpts,nocca,1,nvira)
            Hr2aba = Hr2aba - einsum('JJAjjaa->JAja', imds.WooVV).reshape(nkpts,nkpts,nocca,nvirb,1)

            Hr2bab = Hr2bab + einsum('JBBjbbj,JAB->JAjb', imds.WOVVO, IAB).reshape(nkpts,nkpts,noccb,1,nvirb)
            Hr2bab = Hr2bab - einsum('JJAjjaa->JAja', imds.WOOvv).reshape(nkpts,nkpts,noccb,nvira,1)

            Hr2bbb = Hr2bbb + einsum('JBBjbbj,JAB->JAjb', imds.WOVVO, IAB).reshape(nkpts,nkpts,noccb,1,nvirb)
            Hr2bbb = Hr2bbb + einsum('JAAjaaj->JAja', imds.WOVVO).reshape(nkpts,nkpts,noccb,nvirb,1)

        Hr2aaa = array(Hr2aaa, sym2)
        Hr2aba = array(Hr2aba, sym2)
        Hr2bab = array(Hr2bab, sym2)
        Hr2bbb = array(Hr2bbb, sym2)
        
        return self.amplitudes_to_vector_ea([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb])

    def get_init_guess_ip(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.nip()
        nroots = min(nroots, size)
        nonzero_padding_a, nonzero_padding_b = padding_k_idx(self, kind="split")
        nocca, noccb = self.nocc
        guess = []
        if koopmans:
            idx = np.zeros(nroots, dtype=np.int)
            tmp_oalpha, tmp_obeta = nonzero_padding_a[0][kshift], nonzero_padding_b[0][kshift]
            tmp_oalpha = list(tmp_oalpha)
            tmp_obeta = list(tmp_obeta)
            if len(tmp_obeta) + len(tmp_oalpha) < nroots:
                raise ValueError("Max number of roots for k-point (idx=%3d) for koopmans "
                                 "is %3d.\nRequested %3d." %
                                 (kshift, len(tmp_obeta)+len(tmp_oalpha), nroots))

            total_count = 0
            while(total_count < nroots):
                if total_count % 2 == 0 and len(tmp_oalpha) > 0:
                    idx[total_count] = tmp_oalpha.pop()
                else:
                    # Careful! index depends on how we create vector
                    # (here the first elements are r1a, then r1b)
                    idx[total_count] = nocca + tmp_obeta.pop()
                total_count += 1
        else:
            if diag is None: diag = self.ipccsd_diag(kshift)
            idx = argsort(diag)[:nroots]

        def write_guess(i):
            return idx[i], np.ones(1)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, np.arange(nroots)).array
        return guess

    def get_init_guess_ea(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.nea()
        nroots = min(nroots, size)
        nonzero_padding_a, nonzero_padding_b = padding_k_idx(self, kind="split")
        nocca, noccb= self.nocc
        nmoa, nmob = self.nmo
        nvira, nvirb = nmoa-nocca, nmob-noccb
        guess = []
        if koopmans:
            idx = np.zeros(nroots, dtype=np.int)
            tmp_valpha, tmp_vbeta = nonzero_padding_a[1][kshift], nonzero_padding_b[1][kshift]
            tmp_valpha = list(tmp_valpha)
            tmp_vbeta = list(tmp_vbeta)
            if len(tmp_vbeta) + len(tmp_valpha) < nroots:
                raise ValueError("Max number of roots for k-point (idx=%3d) for koopmans "
                                 "is %3d.\nRequested %3d." %
                                 (kshift, len(tmp_vbeta)+len(tmp_valpha), nroots))

            total_count = 0
            while(total_count < nroots):
                if total_count % 2 == 0 and len(tmp_valpha) > 0:
                    idx[total_count] = tmp_valpha.pop(0)
                else:
                    # Careful! index depends on how we create vector
                    # (here the first elements are r1a, then r1b)
                    idx[total_count] = nvira + tmp_vbeta.pop(0)
                total_count += 1
        else:
            idx = argsort(diag)[:nroots]

        def write_guess(i):
            return idx[i], np.ones(1)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, np.arange(nroots)).array
        return guess

def _make_eris_incore(cc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    eris = uccsd_slow._ChemistsERIs(cc._scf.cell)
    if mo_coeff is None:
        mo_coeff = cc.mo_coeff
    mo_coeff = kccsd_uhf.convert_mo_coeff(mo_coeff)  # FIXME: Remove me!
    mo_coeff = padded_mo_coeff(cc, mo_coeff)
    thisdf = cc._scf.with_df
    cell = cc._scf.cell
    eris.mo_coeff = mo_coeff
    eris.nocc = cc.nocc
    kpts = cc.kpts
    nkpts = cc.nkpts
    nocca, noccb = cc.nocc
    nmoa, nmob = cc.nmo
    nvira, nvirb = nmoa - nocca, nmob - noccb
    sym1, sym2 = cc._sym[:2]
    mo_a, mo_b = mo_coeff
    dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
    if rank==0:
        hcore = cc._scf.get_hcore()
        with lib.temporary_env(cc._scf, exxdiv=None):
            vhf = cc._scf.get_veff(cell, dm)
        focka = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[0][k], mo))
                for k, mo in enumerate(mo_a)]
        fockb = [reduce(np.dot, (mo.conj().T, hcore[k]+vhf[1][k], mo))
                for k, mo in enumerate(mo_b)]
        eris.focka = np.asarray(focka)
        eris.fockb = np.asarray(fockb)
        eris.e_hf = cc._scf.energy_tot(dm=dm, vhf=vhf)

        madelung = tools.madelung(cell, kpts)
        mo_ea = [focka[k].diagonal().real for k in range(nkpts)]
        mo_eb = [fockb[k].diagonal().real for k in range(nkpts)]
        mo_ea = [_adjust_occ(e, nocca, -madelung) for e in mo_ea]
        mo_eb = [_adjust_occ(e, noccb, -madelung) for e in mo_eb]
        eris.mo_energy = (mo_ea, mo_eb)

    ctf_helper.synchronize(eris, ['focka','fockb','e_hf', 'mo_energy'])

    mo_ea_o = [e[:nocca] for e in eris.mo_energy[0]]
    mo_eb_o = [e[:noccb] for e in eris.mo_energy[1]]
    mo_ea_v = [e[nocca:] + cc.level_shift for e in eris.mo_energy[0]]
    mo_eb_v = [e[noccb:] + cc.level_shift for e in eris.mo_energy[1]]

    focka = asarray(eris.focka)
    fockb = asarray(eris.fockb)

    eris.foo = array(focka[:,:nocca,:nocca], sym1)
    eris.fov = array(focka[:,:nocca,nocca:], sym1)
    eris.fvv = array(focka[:,nocca:,nocca:], sym1)

    foo_ = asarray([np.diag(e) for e in mo_ea_o])
    fvv_ = asarray([np.diag(e) for e in mo_ea_v])

    eris._foo = array(foo_, sym1)
    eris._fvv = array(fvv_, sym1)

    eris.fOO = array(fockb[:,:noccb,:noccb], sym1)
    eris.fOV = array(fockb[:,:noccb,noccb:], sym1)
    eris.fVV = array(fockb[:,noccb:,noccb:], sym1)

    fOO_ = asarray([np.diag(e) for e in mo_eb_o])
    fVV_ = asarray([np.diag(e) for e in mo_eb_v])

    eris._fOO = array(fOO_, sym1)
    eris._fVV = array(fVV_, sym1)

    kconserv = cc.khelper.kconserv
    all_tasks = [[ki,kj,ka] for ki,kj,ka in itertools.product(range(nkpts), repeat=3)]
    nonzero_padding_a, nonzero_padding_b = padding_k_idx(cc, kind="split")
    nonzero_opadding_a, nonzero_vpadding_a = nonzero_padding_a
    nonzero_opadding_b, nonzero_vpadding_b = nonzero_padding_b

    eia = np.zeros([nkpts,nocca,nvira])
    eIA = np.zeros([nkpts,noccb,nvirb])
    for ki in range(nkpts):
        eia[ki] =  _get_epq([0,nocca,ki,mo_ea_o,nonzero_opadding_a],
                            [0,nvira,ki,mo_ea_v,nonzero_vpadding_a],
                            fac=[1.0,-1.0])
        eIA[ki] =  _get_epq([0,noccb,ki,mo_eb_o,nonzero_opadding_b],
                            [0,nvirb,ki,mo_eb_v,nonzero_vpadding_b],
                            fac=[1.0,-1.0])
    eris.eia = asarray(eia)
    eris.eIA = asarray(eIA)

    script_mo_a = (nocca, nvira, mo_ea_o, mo_ea_v, nonzero_opadding_a, nonzero_vpadding_a)
    script_mo_b = (noccb, nvirb, mo_eb_o, mo_eb_v, nonzero_opadding_b, nonzero_vpadding_b)

    get_eijab  = lambda ki,kj,ka: kccsd_rhf._get_eijab(ki, kj, ka, kconserv,script_mo_a)
    eris.eijab = frombatchfunc(get_eijab, (nocca,nocca,nvira,nvira), all_tasks, sym=sym2)
    get_eIJAB  = lambda ki,kj,ka: kccsd_rhf._get_eijab(ki, kj, ka, kconserv,script_mo_b)
    eris.eIJAB = frombatchfunc(get_eIJAB, (noccb,noccb,nvirb,nvirb), all_tasks, sym=sym2)
    get_eiJaB  = lambda ki,kj,ka: kccsd_rhf._get_eijab(ki, kj, ka, kconserv,script_mo_a, script_mo_b)
    eris.eiJaB = frombatchfunc(get_eiJaB, (nocca,noccb,nvira,nvirb), all_tasks, sym=sym2)
    from pyscf.pbc import df
    if type(cc._scf.with_df) is df.FFTDF:
        ao2mo.make_fftdf_eris_uhf(cc, eris)
    else:
        from pyscf.ctfcc.integrals import mpigdf
        if type(cc._scf.with_df) is mpigdf.GDF:
            ao2mo.make_df_eris_uhf(cc, eris)
        elif type(cc._scf.with_df) is df.GDF:
            logger.warn(cc, "GDF converted to an MPIGDF object, \
                               one process used for reading from disk")
            cc._scf.with_df = mpigdf.from_serial(cc._scf.with_df)
            ao2mo.make_df_eris_uhf(cc, eris)
        else:
            raise NotImplementedError("DF object not recognized")
    logger.timer(cc, "ao2mo transformation", *cput0)
    return eris

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
    cell.verbose= 4
    cell.build()
    kmf = scf.KUHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
    kmf.kernel()

    mycc = KUCCSD(kmf)
    eris = mycc.ao2mo()
    e, t1, t2 = mycc.kernel(eris=eris)
    print("Energy Error:", e--0.01031579333505543)

    eip, vip = mycc.ipccsd(nroots=3, kptlist=[1])
    print(eip[0,0] - 0.13448793)
    print(eip[0,2] - 0.48273328)
    eea, vea = mycc.eaccsd(nroots=3, kptlist=[2])
    print(eea[0,0] - 1.6094025)
    print(eea[0,2] - 2.22843578)
