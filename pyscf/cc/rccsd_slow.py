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
Restricted CCSD, all integrals/intermediates in memory

Ref: Stanton et al., J. Chem. Phys. 94, 4334 (1990)
Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

minimal memory requirement:
        nocc**4 * 2 + nocc**3*nvir + \
        (nocc*nvir)**2*10 + \
        nocc*nvir**3*2 + nvir**4*2
'''

from functools import reduce
import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd, rccsd, eom_rccsd
from pyscf.cc import rintermediates_slow as imd
from pyscf.lib import linalg_helper

einsum = lib.einsum
dot = np.dot
asarray = np.asarray
eye = np.eye

# note MO integrals are treated in chemist's notation

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)

    Foo = imd.cc_Foo(t1,t2,eris)
    Fvv = imd.cc_Fvv(t1,t2,eris)
    Fov = imd.cc_Fov(t1,t2,eris)

    # Move energy terms to the other side
    Foo -= eris._foo
    Fvv -= eris._fvv

    # T1 equation
    t1new = eris.fov.conj().copy()
    t1new = t1new +-2*einsum('kc,ka,ic->ia', eris.fov, t1, t1)
    t1new +=   einsum('ac,ic->ia', Fvv, t1)
    t1new +=  -einsum('ki,ka->ia', Foo, t1)
    t1new += 2*einsum('kc,kica->ia', Fov, t2)
    t1new +=  -einsum('kc,ikca->ia', Fov, t2)
    t1new +=   einsum('kc,ic,ka->ia', Fov, t1, t1)
    t1new += 2*einsum('kcai,kc->ia', eris.ovvo, t1)
    t1new +=  -einsum('kiac,kc->ia', eris.oovv, t1)

    t1new += 2*einsum('kdac,ikcd->ia', eris.ovvv, t2)
    t1new +=  -einsum('kcad,ikcd->ia', eris.ovvv, t2)
    t1new += 2*einsum('kdac,kd,ic->ia', eris.ovvv, t1, t1)
    t1new +=  -einsum('kcad,kd,ic->ia', eris.ovvv, t1, t1)
    t1new +=-2*einsum('kilc,klac->ia', eris.ooov, t2)
    t1new +=   einsum('likc,klac->ia', eris.ooov, t2)
    t1new +=-2*einsum('kilc,lc,ka->ia', eris.ooov, t1, t1)
    t1new +=   einsum('likc,lc,ka->ia', eris.ooov, t1, t1)

    # T2 equation
    t2new = eris.ovov.copy().conj().transpose(0,2,1,3)
    if cc.cc2:
        Woooo2 = eris.oooo.copy().transpose(0,2,1,3)
        Woooo2 += einsum('kilc,jc->klij', eris.ooov, t1)
        Woooo2 += einsum('ljkc,ic->klij', eris.ooov, t1)
        Woooo2 += einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
        t2new += einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = einsum('kcbd,ka->abcd', eris.ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1,0,3,2)
        Wvvvv += eris.vvvv.transpose(0,2,1,3)
        t2new += einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = eris.fvv - einsum('kc,ka->ac', eris.fov, t1)
        Lvv2 -= eris._fvv
        tmp = einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        Loo2 = eris.foo + einsum('kc,ic->ki', eris.fov, t1)
        Loo2 -= eris._foo
        tmp = einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo -= eris._foo
        Lvv -= eris._fvv
        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)
        Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
        tau = t2 + einsum('ia,jb->ijab', t1, t1)
        t2new = t2new + einsum('klij,klab->ijab', Woooo, tau)
        t2new += einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -=   einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = einsum('kibc,ka->abic', eris.oovv, -t1)
    tmp2 += eris.ovvv.conj().transpose(1,3,0,2)
    tmp = einsum('abic,jc->ijab', tmp2, t1)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris.ooov.transpose(3,1,2,0).conj()
    tmp = einsum('akij,kb->ijab', tmp2, t1)
    t2new -= (tmp + tmp.transpose(1,0,3,2))
    t1new /= eris.eia
    t2new /= eris.eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    e = 2*einsum('ia,ia', eris.fov, t1)
    tau = einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*einsum('ijab,iajb', tau, eris.ovov)
    e +=  -einsum('ijab,ibja', tau, eris.ovov)
    return e.real

def make_intermediates(mycc, t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.foo
    fov = eris.fov
    fvo = eris.fov.conj().transpose(1,0)
    fvv = eris.fvv

    tau = t2 + einsum('ia,jb->ijab', t1, t1)

    ovov = eris.ovov
    ovoo = eris.ooov.transpose(2,3,0,1)
    ovov1 = ovov * 2 - ovov.transpose(0,3,2,1)
    ovoo1 = ovoo * 2 - ovoo.transpose(2,1,0,3)

    v1  = fvv - einsum('ja,jb->ba', fov, t1)
    v1 -= einsum('jakc,jkbc->ba', ovov1, tau)
    v2  = foo + einsum('ib,jb->ij', fov, t1)
    v2 += einsum('ibkc,jkbc->ij', ovov1, tau)
    v2 += einsum('kbij,kb->ij', ovoo1, t1)
    v4 = fov + einsum('jbkc,kc->jb', ovov1, t1)

    v5  = einsum('kc,jkbc->bj', fov, t2) * 2
    v5 -= einsum('kc,jkcb->bj', fov, t2)
    v5 += fvo
    v5 += einsum('kc,kb,jc->bj', v4, t1, t1)
    v5 -= einsum('lckj,klbc->bj', ovoo1, t2)

    oooo = eris.oooo
    woooo  = einsum('icjl,kc->ikjl', ovoo, t1)
    woooo += einsum('jcil,kc->iljk', ovoo, t1)
    woooo += oooo
    woooo += einsum('icjd,klcd->ikjl', ovov, tau)

    theta = t2*2 - t2.transpose(0,1,3,2)
    v4OVvo  = einsum('ldjb,klcd->jbck', ovov1, t2)
    v4OVvo -= einsum('ldjb,kldc->jbck', ovov, t2)
    v4OVvo += eris.ovvo

    v4oVVo  = einsum('jdlb,kldc->jbck', ovov, t2)
    v4oVVo -= eris.oovv.transpose(0,3,2,1)

    v4ovvo = v4OVvo*2 + v4oVVo
    w3 = einsum('jbck,jb->ck', v4ovvo, t1)

    woovo  = einsum('ibck,jb->ijck', v4ovvo, t1)
    woovo = woovo - woovo.transpose(0,3,2,1)
    woovo += einsum('ibck,jb->ikcj', v4OVvo-v4oVVo, t1)
    woovo += ovoo1.conj().transpose(3,2,1,0)

    woovo += einsum('lcik,jlbc->ikbj', ovoo1, theta)
    woovo -= einsum('lcik,jlbc->ijbk', ovoo1, t2)
    woovo -= einsum('iclk,ljbc->ijbk', ovoo1, t2)

    wvvvo  = einsum('jack,jb->back', v4ovvo, t1)
    wvvvo = wvvvo - wvvvo.transpose(2,1,0,3)
    wvvvo += einsum('jack,jb->cabk', v4OVvo-v4oVVo, t1)
    wvvvo -= einsum('lajk,jlbc->cabk', ovoo1, tau)

    wOVvo  = v4OVvo
    woVVo  = v4oVVo
    wOVvo -= einsum('jbld,kd,lc->jbck', ovov, t1, t1)
    woVVo += einsum('jdlb,kd,lc->jbck', ovov, t1, t1)
    wOVvo -= einsum('jblk,lc->jbck', ovoo, t1)
    woVVo += einsum('lbjk,lc->jbck', ovoo, t1)
    v4ovvo = v4OVvo = v4oVVo = None

    ovvv = eris.ovvv
    wvvvo += einsum('kacd,kjbd->bacj', ovvv, t2) * 1.5

    wOVvo += einsum('jbcd,kd->jbck', ovvv, t1)
    woVVo -= einsum('jdcb,kd->jbck', ovvv, t1)

    ovvv = ovvv*2 - ovvv.transpose(0,3,2,1)
    v1 += einsum('jcba,jc->ba', ovvv, t1)
    v5 += einsum('kdbc,jkcd->bj', ovvv, t2)
    woovo += einsum('idcb,jkdb->ijck', ovvv, tau)

    tmp = einsum('kdca,jkbd->cabj', ovvv, theta)
    wvvvo -= tmp
    wvvvo += tmp.transpose(2,1,0,3) * .5
    wvvvo -= ovvv.conj().transpose(3,2,1,0)
    ovvv = tmp = None

    w3 += v5
    w3 += einsum('cb,jb->cj', v1, t1)
    w3 -= einsum('jk,jb->bk', v2, t1)

    class _LIMDS: pass
    imds = _LIMDS()
    imds.woooo = woooo
    imds.wovvo = wOVvo*2 + woVVo
    imds.woVVo = woVVo
    imds.woovo = woovo
    imds.wvvvo = wvvvo
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.v4 = v4
    return imds

def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape

    foo = eris.foo
    fov = eris.fov
    fvv = eris.fvv

    tau = t2 + einsum('ia,jb->ijab', t1, t1)

    theta = t2*2 - t2.transpose(0,1,3,2)
    mvv = einsum('klca,klcb->ba', l2, theta)
    moo = einsum('kicd,kjcd->ij', l2, theta)
    mvv1 = einsum('jc,jb->bc', l1, t1) + mvv
    moo1 = einsum('ic,kc->ik', l1, t1) + moo

    m3 = einsum('ijab,acbd->ijcd', l2, eris.vvvv)
    m3 += einsum('klab,ikjl->ijab', l2, imds.woooo)
    m3 *= .5

    ovov = eris.ovov
    l2tau = einsum('ijcd,klcd->ijkl', l2, tau)
    m3 += einsum('kalb,ijkl->ijab', ovov, l2tau) * .5
    l2tau = None

    l2new = ovov.transpose(0,2,1,3) * .5
    l2new = l2new + einsum('ijac,cb->ijab', l2, imds.v1)
    l2new -= einsum('ikab,jk->ijab', l2, imds.v2)
    l2new -= einsum('ca,icjb->ijab', mvv1, ovov)
    l2new -= einsum('ik,kajb->ijab', moo1, ovov)

    ovov = ovov * 2 - ovov.transpose(0,3,2,1)
    l1new = - einsum('ik,ka->ia', moo, imds.v4)
    l1new -= einsum('ca,ic->ia', mvv, imds.v4)
    l2new += einsum('ia,jb->ijab', l1, imds.v4)

    tmp  = t1 + einsum('kc,kjcb->jb', l1, theta)
    tmp -= einsum('bd,jd->jb', mvv1, t1)
    tmp -= einsum('lj,lb->jb', moo, t1)
    l1new += einsum('jbia,jb->ia', ovov, tmp)
    ovov = tmp = None

    ovvv = eris.ovvv
    l1new += einsum('iacb,bc->ia', ovvv, mvv1) * 2
    l1new -= einsum('ibca,bc->ia', ovvv, mvv1)
    l2new += einsum('ic,jbca->jiba', l1, ovvv)
    l2t1 = einsum('ijcd,kd->ijck', l2, t1)
    m3 -= einsum('kbca,ijck->ijab', ovvv, l2t1)
    l2t1 = ovvv = None

    l2new += m3
    l1new += einsum('ijab,jb->ia', m3, t1) * 2
    l1new += einsum('jiba,jb->ia', m3, t1) * 2
    l1new -= einsum('ijba,jb->ia', m3, t1)
    l1new -= einsum('jiab,jb->ia', m3, t1)

    ovoo = eris.ooov.transpose(2,3,0,1)
    l1new -= einsum('iajk,kj->ia', ovoo, moo1) * 2
    l1new += einsum('jaik,kj->ia', ovoo, moo1)
    l2new -= einsum('ka,jbik->ijab', l1, ovoo)
    ovoo = None

    l2theta = l2*2 - l2.transpose(0,1,3,2)
    l2new += einsum('ikac,jbck->ijab', l2theta, imds.wovvo) * .5
    tmp = einsum('ikca,jbck->ijab', l2, imds.woVVo)
    l2new += tmp * .5
    l2new += tmp.transpose(1,0,2,3)
    l2theta = None

    l1new += fov
    l1new += einsum('ib,ba->ia', l1, imds.v1)
    l1new -= einsum('ja,ij->ia', l1, imds.v2)

    l1new += einsum('jb,iabj->ia', l1, eris.ovvo) * 2
    l1new -= einsum('jb,ijba->ia', l1, eris.oovv)

    l1new -= einsum('ijbc,bacj->ia', l2, imds.wvvvo)
    l1new -= einsum('kjca,ijck->ia', l2, imds.woovo)

    l1new += einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= einsum('ijba,bj->ia', l2, imds.w3)

    l1new /= eris.eia
    l1new += l1

    l2new = l2new + l2new.transpose(1,0,3,2)
    l2new /= eris.eijab
    l2new += l2

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


class RCCSD(rccsd.RCCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        rccsd.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self.ip_partition = self.ea_partition = None
        self._keys = self._keys.union(['max_space', 'ip_partition', 'ea_partition'])

    def init_amps(self, eris):
        t1 = eris.fov.conj() / eris.eia
        t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
        self.emp2  = 2*einsum('ijab,iajb', t2, eris.ovov)
        self.emp2 -=   einsum('ijab,ibja', t2, eris.ovov)
        lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2.real)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    energy = energy
    update_amps = update_amps

    nip = eom_rccsd.EOMIP.vector_size
    nea = eom_rccsd.EOMEA.vector_size
    nee = eom_rccsd.EOMEE.vector_size

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose,
                                    fintermediates=make_intermediates, fupdate=update_lambda)
        return self.l1, self.l2

    def ccsd_t_slow(self, t1=None, t2=None, eris=None):
        if eris is None: eris = self.ao2mo()
        if t1 is None or t2 is None:
            t1, t2 = self.t1, self.t2

        nocc, nvir  = t1.shape
        eris_vvov = eris.ovvv.conj().transpose(1,3,0,2)
        eris_vooo = eris.ooov.conj().transpose(3,2,1,0)
        eris_vvoo = eris.ovov.conj().transpose(1,3,0,2)
        fvo = eris.fov.conj().transpose(1,0)

        w = einsum('abif,kjcf->ijkabc', eris_vvov, t2)
        w-= einsum('aijm,mkbc->ijkabc', eris_vooo, t2)

        pw = w + w.transpose(2,0,1,5,3,4) + \
             w.transpose(1,2,0,4,5,3) + w.transpose(0,2,1,3,5,4) + \
             w.transpose(2,1,0,5,4,3)+ w.transpose(1,0,2,4,3,5)

        rw = 4.*pw + pw.transpose(0,1,2,5,3,4) + \
             pw.transpose(0,1,2,4,5,3) - 2.* pw.transpose(0,1,2,3,5,4) - \
             2.*pw.transpose(0,1,2,5,4,3) - 2.*pw.transpose(0,1,2,4,3,5)


        v = einsum('abij,kc->ijkabc', eris_vvoo, t1)
        v+= einsum('ijab,ck->ijkabc', t2, fvo)

        pv = v + v.transpose(2,0,1,5,3,4) + \
             v.transpose(1,2,0,4,5,3) + v.transpose(0,2,1,3,5,4) + \
             v.transpose(2,1,0,5,4,3)+ v.transpose(1,0,2,4,3,5)
        v = None
        pw = pw + .5*pv
        pv = None

        d3 = eris.get_eijkabc()

        energy_t = einsum('abcijk,abcijk', pw, rw.conj()/d3) / 3.
        return energy_t

    def ccsd_t(self, t1=None, t2=None, eris=None, slice_size=None, free_vvvv=False):
        if eris is None: eris = self.ao2mo()
        if free_vvvv: eris.vvvv = None
        if t1 is None or t2 is None:
            t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape
        if slice_size is None:
            slice_size = nvir**4 / 18
        else:
            slice_size = slice_size / 18 * 1.25e5

        vir_blksize = min(nvir, max(4, int(((slice_size)**(1.0/3)/nocc))))
        logger.info(self, "nvir=%i, virtual blksize=%i", nvir, vir_blksize)
        if vir_blksize == nvir:
            return self.ccsd_t_slow(t1, t2, eris)

        tasks = []
        for a0, a1 in lib.prange(0, nvir, vir_blksize):
            for b0, b1 in lib.prange(0, nvir, vir_blksize):
                for c0, c1 in lib.prange(0, nvir, vir_blksize):
                    if b0>=a0 and c0>=b0:
                        tasks.append((a0,a1,b0,b1,c0,c1))

        e_occ = eris.foo.diagonal()
        e_vir = eris.fvv.diagonal()

        eijk = e_occ.reshape(nocc,1,1) + e_occ.reshape(1,nocc,1) + e_occ.reshape(1,1,nocc)

        eris_vvov = eris.ovvv.conj().transpose(1,3,0,2)
        eris_vooo = eris.ooov.conj().transpose(3,2,1,0)
        eris_vvoo = eris.ovov.conj().transpose(1,3,0,2)
        fvo = eris.fov.conj().transpose(1,0)

        def get_w(a0, a1, b0, b1, c0, c1):
            w = einsum('abif,kjcf->ijkabc', eris_vvov[a0:a1,b0:b1], t2[:,:,c0:c1])
            w-= einsum('aijm,mkbc->ijkabc', eris_vooo[a0:a1,:], t2[:,:,b0:b1,c0:c1])
            return w

        def get_v(a0, a1, b0, b1, c0, c1):
            v = einsum('abij,kc->ijkabc', eris_vvoo[a0:a1,b0:b1], t1[:,c0:c1])
            v+= einsum('ijab,ck->ijkabc', t2[:,:,a0:a1,b0:b1], fvo[c0:c1])
            return v

        def r3(w):
            return (4 * w + w.transpose(1,2,0,3,4,5) + w.transpose(2,0,1,3,4,5)
                    - 2 * w.transpose(2,1,0,3,4,5) - 2 * w.transpose(0,2,1,3,4,5)
                    - 2 * w.transpose(1,0,2,3,4,5))

        et = 0
        for (a0, a1, b0, b1, c0, c1) in tasks:
            d3 = eijk.reshape(nocc,nocc,nocc,1,1,1) - e_vir[a0:a1].reshape(1,1,1,-1,1,1)\
                -e_vir[b0:b1].reshape(1,1,1,1,-1,1) - e_vir[c0:c1].reshape(1,1,1,1,1,-1)
            if a0 == c0:
                d3 *= 6
            elif a0 == b0 or b0 == c0:
                d3 *= 2

            wabc = get_w(a0, a1, b0, b1, c0, c1)
            wacb = get_w(a0, a1, c0, c1, b0, b1)
            wbac = get_w(b0, b1, a0, a1, c0, c1)
            wbca = get_w(b0, b1, c0, c1, a0, a1)
            wcab = get_w(c0, c1, a0, a1, b0, b1)
            wcba = get_w(c0, c1, b0, b1, a0, a1)
            vabc = get_v(a0, a1, b0, b1, c0, c1)
            vacb = get_v(a0, a1, c0, c1, b0, b1)
            vbac = get_v(b0, b1, a0, a1, c0, c1)
            vbca = get_v(b0, b1, c0, c1, a0, a1)
            vcab = get_v(c0, c1, a0, a1, b0, b1)
            vcba = get_v(c0, c1, b0, b1, a0, a1)

            zabc = r3(wabc + .5 * vabc) / d3
            zacb = r3(wacb + .5 * vacb) / d3.transpose(0,1,2,3,5,4)
            zbac = r3(wbac + .5 * vbac) / d3.transpose(0,1,2,4,3,5)
            zbca = r3(wbca + .5 * vbca) / d3.transpose(0,1,2,4,5,3)
            zcab = r3(wcab + .5 * vcab) / d3.transpose(0,1,2,5,3,4)
            zcba = r3(wcba + .5 * vcba) / d3.transpose(0,1,2,5,4,3)

            et+= einsum('ijkabc,ijkabc', wabc, zabc.conj())
            et+= einsum('ikjacb,ijkabc', wacb, zabc.conj())
            et+= einsum('jikbac,ijkabc', wbac, zabc.conj())
            et+= einsum('jkibca,ijkabc', wbca, zabc.conj())
            et+= einsum('kijcab,ijkabc', wcab, zabc.conj())
            et+= einsum('kjicba,ijkabc', wcba, zabc.conj())

            et+= einsum('ijkacb,ijkacb', wacb, zacb.conj())
            et+= einsum('ikjabc,ijkacb', wabc, zacb.conj())
            et+= einsum('jikcab,ijkacb', wcab, zacb.conj())
            et+= einsum('jkicba,ijkacb', wcba, zacb.conj())
            et+= einsum('kijbac,ijkacb', wbac, zacb.conj())
            et+= einsum('kjibca,ijkacb', wbca, zacb.conj())

            et+= einsum('ijkbac,ijkbac', wbac, zbac.conj())
            et+= einsum('ikjbca,ijkbac', wbca, zbac.conj())
            et+= einsum('jikabc,ijkbac', wabc, zbac.conj())
            et+= einsum('jkiacb,ijkbac', wacb, zbac.conj())
            et+= einsum('kijcba,ijkbac', wcba, zbac.conj())
            et+= einsum('kjicab,ijkbac', wcab, zbac.conj())

            et+= einsum('ijkbca,ijkbca', wbca, zbca.conj())
            et+= einsum('ikjbac,ijkbca', wbac, zbca.conj())
            et+= einsum('jikcba,ijkbca', wcba, zbca.conj())
            et+= einsum('jkicab,ijkbca', wcab, zbca.conj())
            et+= einsum('kijabc,ijkbca', wabc, zbca.conj())
            et+= einsum('kjiacb,ijkbca', wacb, zbca.conj())

            et+= einsum('ijkcab,ijkcab', wcab, zcab.conj())
            et+= einsum('ikjcba,ijkcab', wcba, zcab.conj())
            et+= einsum('jikacb,ijkcab', wacb, zcab.conj())
            et+= einsum('jkiabc,ijkcab', wabc, zcab.conj())
            et+= einsum('kijbca,ijkcab', wbca, zcab.conj())
            et+= einsum('kjibac,ijkcab', wbac, zcab.conj())

            et+= einsum('ijkcba,ijkcba', wcba, zcba.conj())
            et+= einsum('ikjcab,ijkcba', wcab, zcba.conj())
            et+= einsum('jikbca,ijkcba', wbca, zcba.conj())
            et+= einsum('jkibac,ijkcba', wbac, zcba.conj())
            et+= einsum('kijacb,ijkcba', wacb, zcba.conj())
            et+= einsum('kjiabc,ijkcba', wabc, zcba.conj())

        et*= 2.0
        logger.info(self, 'CCSD(T) correction = %.15g', et)
        return et

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nip()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        adiag = self.ipccsd_diag()
        if partition == 'full':
            self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(adiag)[1]
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[self.nocc-n-1] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        if left:
            matvec = self.lipccsd_matvec
        else:
            matvec = self.ipccsd_matvec
        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eip, evecs = eig(matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eip, evecs = eig(matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eip = eip.real

        if nroots == 1:
            eip, evecs = [self.eip], [evecs]
        for n, en, vn in zip(range(nroots), eip, evecs):
            logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:self.nocc])**2)
        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def ipccsd_matvec(self, vector, **kwargs):
        # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ip(vector, **kwargs)

        # 1h-1h block
        Hr1 = -einsum('ki,k->i',imds.Loo,r1)
        #1h-2h1p block
        Hr1 += 2*einsum('ld,ild->i',imds.Fov,r2)
        Hr1 +=  -einsum('kd,kid->i',imds.Fov,r2)
        Hr1 += -2*einsum('klid,kld->i',imds.Wooov,r2)
        Hr1 +=    einsum('lkid,kld->i',imds.Wooov,r2)

        # 2h1p-1h block
        Hr2 = -einsum('kbij,k->ijb',imds.Wovoo,r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            fvv = imds.eris.fvv
            foo = imds.eris.foo
            Hr2 += einsum('bd,ijd->ijb',fvv,r2)
            Hr2 += -einsum('ki,kjb->ijb',foo,r2)
            Hr2 += -einsum('lj,ilb->ijb',foo,r2)
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('bd,ijd->ijb',imds.Lvv,r2)
            Hr2 += -einsum('ki,kjb->ijb',imds.Loo,r2)
            Hr2 += -einsum('lj,ilb->ijb',imds.Loo,r2)
            Hr2 +=  einsum('klij,klb->ijb',imds.Woooo,r2)
            Hr2 += 2*einsum('lbdj,ild->ijb',imds.Wovvo,r2)
            Hr2 +=  -einsum('kbdj,kid->ijb',imds.Wovvo,r2)
            Hr2 +=  -einsum('lbjd,ild->ijb',imds.Wovov,r2) #typo in Ref
            Hr2 +=  -einsum('kbid,kjd->ijb',imds.Wovov,r2)
            tmp = 2*einsum('lkdc,kld->c',imds.Woovv,r2)
            tmp += -einsum('kldc,kld->c',imds.Woovv,r2)
            Hr2 += -einsum('c,ijcb->ijb',tmp,imds.t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2,**kwargs)
        return vector

    def lipccsd_matvec(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds
        t2 = imds.t2

        r1,r2 = self.vector_to_amplitudes_ip(vector, **kwargs)

        # 1h-1h block
        Hr1 = -einsum('ki,i->k',imds.Loo,r1)
        #1h-2h1p block
        Hr1 += -einsum('kbij,ijb->k',imds.Wovoo,r2)

        # 2h1p-1h block
        Hr2 = -einsum('kd,l->kld',imds.Fov,r1)
        Hr2 += 2.*einsum('ld,k->kld',imds.Fov,r1)
        Hr2 += -2.*einsum('klid,i->kld',imds.Wooov,r1)
        Hr2 += einsum('lkid,i->kld',imds.Wooov,r1)
        # 2h1p-2h1p block
        if self.ip_partition == 'mp':
            foo = imds.eris.foo
            fvv = imds.eris.fvv
            Hr2 += einsum('bd,klb->kld', fvv,r2)
            Hr2 += -einsum('ki,ild->kld', foo,r2)
            Hr2 += -einsum('lj,kjd->kld', foo,r2)
        elif self.ip_partition == 'full':
            Hr2 += self._ipccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('bd,klb->kld',imds.Lvv,r2)
            Hr2 += -einsum('ki,ild->kld',imds.Loo,r2)
            Hr2 += -einsum('lj,kjd->kld',imds.Loo,r2)
            Hr2 += 2.*einsum('lbdj,kjb->kld',imds.Wovvo,r2)
            Hr2 += -einsum('kbdj,ljb->kld',imds.Wovvo,r2)
            Hr2 += -einsum('lbjd,kjb->kld',imds.Wovov,r2)
            Hr2 += einsum('klij,ijd->kld',imds.Woooo,r2)
            Hr2 += -einsum('kbid,ilb->kld',imds.Wovov,r2)
            tmp = einsum('ijcb,ijb->c',t2,r2)
            Hr2 += einsum('kldc,c->kld',imds.Woovv,tmp)
            Hr2 += -2.*einsum('lkdc,c->kld',imds.Woovv,tmp)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2,**kwargs)
        return vector

    def ipccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape
        foo = imds.eris.foo
        fvv = imds.eris.fvv
        Hr1 = -imds.Loo.diagonal()
        if self.ip_partition == 'mp':
            Hr2 = fvv.diagonal().reshape(1,1,nvir) \
                - foo.diagonal().reshape(1,nocc,1) \
                - foo.diagonal().reshape(nocc,1,1)
        else:
            Hr2 = imds.Lvv.diagonal().reshape(1,1,nvir) \
                - imds.Loo.diagonal().reshape(1,nocc,1) \
                - imds.Loo.diagonal().reshape(nocc,1,1)

            Hr2 = Hr2 + einsum('ijij->ij', imds.Woooo).reshape(nocc,nocc,1)
            Hr2 = Hr2 + 2*einsum('jbbj->jb', imds.Wovvo).reshape(1,nocc,nvir)
            Hr2 = Hr2 - einsum('ibbi,ij->ijb', imds.Wovvo, eye(nocc))
            Hr2 = Hr2 - einsum('jbjb->jb', imds.Wovov).reshape(1,nocc,nvir)
            Hr2 = Hr2 - einsum('ibib->ib', imds.Wovov).reshape(nocc,1,nvir)
            Hr2 = Hr2 - 2*einsum('jibe,ijeb->ijb', imds.Woovv, t2)
            Hr2 = Hr2 + einsum('ijbe,ijeb->ijb', imds.Woovv, t2)

        vector = self.amplitudes_to_vector_ip(Hr1,Hr2)
        return vector

    def vector_to_amplitudes_ip(self, vector, **kwargs):
        return eom_rccsd.EOMIP.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_ea(self, vector, **kwargs):
        return eom_rccsd.EOMEA.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_singlet(self, vector, **kwargs):
        return eom_rccsd.EOMEESinglet.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_triplet(self, vector, **kwargs):
        return eom_rccsd.EOMEETriplet.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_eomsf(self, vector, **kwargs):
        return eom_rccsd.EOMEESpinFlip.vector_to_amplitudes(self, vector)

    amplitudes_to_vector_ip = eom_rccsd.EOMIP.amplitudes_to_vector
    amplitudes_to_vector_ea = eom_rccsd.EOMEA.amplitudes_to_vector
    amplitudes_to_vector_singlet = eom_rccsd.EOMEESinglet.amplitudes_to_vector
    amplitudes_to_vector_triplet = eom_rccsd.EOMEETriplet.amplitudes_to_vector
    amplitudes_to_vector_eomsf = eom_rccsd.EOMEESpinFlip.amplitudes_to_vector

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, **kwargs):
        assert(self.ip_partition == None)

        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        eris = self.imds.eris

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc

        foo = eris.foo
        fvv = eris.fvv

        ovov = eris.ovov
        ovvv = eris.ovvv
        vovv = ovvv.conj().transpose(1,0,3,2)
        oovv = eris.oovv
        ovvo = eris.ovvo
        ooov = eris.ooov
        vooo = ooov.conj().transpose(3,2,1,0)
        oooo = eris.oooo

        eijkab = eris.get_eijkab(**kwargs)

        e = []
        for _eval, _evec, _levec in zip(ipccsd_evals, ipccsd_evecs, lipccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ip(_levec, **kwargs)
            r1,r2 = self.vector_to_amplitudes_ip(_evec, **kwargs)
            ldotr = dot(l1.conj(),r1.ravel()) + dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

            _eijkab = eijkab + _eval
            _eijkab = 1./_eijkab

            lijkab = 0.5*einsum('iajb,k->ijkab', ovov, l1)
            lijkab += einsum('iaeb,jke->ijkab', ovvv, l2)
            lijkab += -einsum('kmjb,ima->ijkab', ooov, l2)
            lijkab += -einsum('imjb,mka->ijkab', ooov, l2)
            lijkab = lijkab + lijkab.transpose(1,0,2,4,3)

            rijkab = -einsum('mkbe,m,ijae->ijkab', oovv, r1, t2)
            rijkab -= einsum('mebj,m,ikae->ijkab', ovvo, r1, t2)
            rijkab += einsum('mjnk,n,imab->ijkab', oooo, r1, t2)
            rijkab +=  einsum('aibe,kje->ijkab', vovv, r2)
            rijkab += -einsum('bjmk,mia->ijkab', vooo, r2)
            rijkab += -einsum('bjmi,kma->ijkab', vooo, r2)
            rijkab = rijkab + rijkab.transpose(1,0,2,4,3)

            lijkab = 4.*lijkab \
                   - 2.*lijkab.transpose(1,0,2,3,4) \
                   - 2.*lijkab.transpose(2,1,0,3,4) \
                   - 2.*lijkab.transpose(0,2,1,3,4) \
                   + 1.*lijkab.transpose(1,2,0,3,4) \
                   + 1.*lijkab.transpose(2,0,1,3,4)

            deltaE = 0.5*einsum('ijkab,ijkab',lijkab,rijkab*_eijkab)
            deltaE = deltaE.real
            logger.note(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccsd()
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nea()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        adiag = self.eaccsd_diag()
        if partition == 'full':
            self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(adiag)[1]
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            if koopmans:
                for n in range(nroots):
                    g = np.zeros(size)
                    g[n] = 1.0
                    guess.append(g)
            else:
                idx = adiag.argsort()[:nroots]
                for i in idx:
                    g = np.zeros(size)
                    g[i] = 1.0
                    guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        if left:
            matvec = self.leaccsd_matvec
        else:
            matvec = self.eaccsd_matvec
        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eea, evecs = eig(matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eea, evecs = eig(matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eea = eea.real

        if nroots == 1:
            eea, evecs = [self.eea], [evecs]
        nvir = self.nmo - self.nocc
        for n, en, vn in zip(range(nroots), eea, evecs):
            logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:nvir])**2)
        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def eaccsd_matvec(self,vector, **kwargs):
        # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        r1,r2 = self.vector_to_amplitudes_ea(vector, **kwargs)

        # Eq. (30)
        # 1p-1p block
        Hr1 =  einsum('ac,c->a',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('ld,lad->a',2.*imds.Fov,r2)
        Hr1 += einsum('ld,lda->a',  -imds.Fov,r2)
        Hr1 += 2*einsum('alcd,lcd->a',imds.Wvovv,r2)
        Hr1 +=  -einsum('aldc,lcd->a',imds.Wvovv,r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = einsum('abcj,c->jab',imds.Wvvvo,r1)
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            foo = imds.eris.foo
            fvv = imds.eris.fvv
            Hr2 +=  einsum('ac,jcb->jab',fvv,r2)
            Hr2 +=  einsum('bd,jad->jab',fvv,r2)
            Hr2 += -einsum('lj,lab->jab',foo,r2)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 +=  einsum('ac,jcb->jab',imds.Lvv,r2)
            Hr2 +=  einsum('bd,jad->jab',imds.Lvv,r2)
            Hr2 += -einsum('lj,lab->jab',imds.Loo,r2)

            Hr2 += 2*einsum('lbdj,lad->jab',imds.Wovvo,r2)
            Hr2 +=  -einsum('lbjd,lad->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lajc,lcb->jab',imds.Wovov,r2)
            Hr2 +=  -einsum('lbcj,lca->jab',imds.Wovvo,r2)

            Hr2 += einsum('abcd,jcd->jab', imds.Wvvvv, r2)
            tmp = (2*einsum('klcd,lcd->k',imds.Woovv,r2)
                    -einsum('kldc,lcd->k',imds.Woovv,r2))
            Hr2 += -einsum('k,kjab->jab',tmp,imds.t2)


        vector = self.amplitudes_to_vector_ea(Hr1,Hr2, **kwargs)
        return vector

    def leaccsd_matvec(self,vector, **kwargs):
        # Note this is not the same left EA equations used by Nooijen and Bartlett.
        # Small changes were made so that the same type L2 basis was used for both the
        # left EA and left IP equations.  You will note more similarity for these
        # equations to the left IP equations than for the left EA equations by Nooijen.
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds
        t2 = imds.t2
        r1,r2 = self.vector_to_amplitudes_ea(vector, **kwargs)

        # Eq. (30)
        # 1p-1p block
        Hr1 = einsum('ac,a->c',imds.Lvv,r1)
        # 1p-2p1h block
        Hr1 += einsum('abcj,jab->c',imds.Wvvvo,r2)
        # Eq. (31)
        # 2p1h-1p block
        Hr2 = 2.*einsum('c,ld->lcd',r1,imds.Fov)
        Hr2 +=   -einsum('d,lc->lcd',r1,imds.Fov)
        Hr2 += 2.*einsum('a,alcd->lcd',r1,imds.Wvovv)
        Hr2 +=   -einsum('a,aldc->lcd',r1,imds.Wvovv)
        # 2p1h-2p1h block
        if self.ea_partition == 'mp':
            foo = imds.eris.foo
            fvv = imds.eris.fvv
            Hr2 += einsum('lad,ac->lcd',r2,fvv)
            Hr2 += einsum('lcb,bd->lcd',r2,fvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,foo)
        elif self.ea_partition == 'full':
            Hr2 += self._eaccsd_diag_matrix2*r2
        else:
            Hr2 += einsum('lad,ac->lcd',r2,imds.Lvv)
            Hr2 += einsum('lcb,bd->lcd',r2,imds.Lvv)
            Hr2 += -einsum('jcd,lj->lcd',r2,imds.Loo)
            Hr2 += 2.*einsum('jcb,lbdj->lcd',r2,imds.Wovvo)
            Hr2 +=   -einsum('jcb,lbjd->lcd',r2,imds.Wovov)
            Hr2 +=   -einsum('lajc,jad->lcd',imds.Wovov,r2)
            Hr2 +=   -einsum('lbcj,jdb->lcd',imds.Wovvo,r2)
            Hr2 += einsum('lab,abcd->lcd',r2,imds.Wvvvv)
            tmp = einsum('ijcb,ibc->j',t2,r2)
            Hr2 +=     einsum('kjef,j->kef',imds.Woovv,tmp)
            Hr2 += -2.*einsum('kjfe,j->kef',imds.Woovv,tmp)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2,**kwargs)
        return vector

    def eaccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds

        t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        foo = imds.eris.foo
        fvv = imds.eris.fvv

        Hr1 = imds.Lvv.diagonal()
        Hr2 = np.zeros((nocc,nvir,nvir), t1.dtype)
        if self.ea_partition == 'mp':
            Hr2 = fvv.diagonal().reshape(1,nvir,1) +\
                  fvv.diagonal().reshape(1,1,nvir) -\
                  foo.diagonal().reshape(nocc,1,1)
        else:
            Hr2 = imds.Lvv.diagonal().reshape(1,nvir,1) +\
                  imds.Lvv.diagonal().reshape(1,1,nvir) -\
                  imds.Loo.diagonal().reshape(nocc,1,1)
            Hr2 = Hr2 + 2*einsum('jbbj->jb', imds.Wovvo).reshape(nocc,1,nvir)
            Hr2 = Hr2 - einsum('jbjb->jb', imds.Wovov).reshape(nocc,1,nvir)
            Hr2 = Hr2 - einsum('jaja->ja', imds.Wovov).reshape(nocc,nvir,1)
            Hr2 = Hr2 - einsum('jbbj,ab->jab', imds.Wovvo, eye(nvir))
            Hr2 = Hr2 + einsum('abab->ab', imds.Wvvvv).reshape(1,nvir,nvir)
            Hr2 = Hr2 - 2*einsum('ijab,ijab->jab', imds.Woovv, t2)
            Hr2 = Hr2 + einsum('ijba,ijab->jab', imds.Woovv, t2)
        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, **kwargs):
        assert(self.ea_partition == None)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        eris = self.imds.eris

        t1, t2 = self.t1, self.t2
        nocc = self.nocc
        nvir = self.nmo - nocc

        foo = eris.foo
        fvv = eris.fvv

        ovov = eris.ovov.copy()
        ovvv = eris.ovvv.copy()
        vovv = ovvv.conj().transpose(1,0,3,2)
        ooov = eris.ooov.copy()
        vooo = ooov.conj().transpose(3,2,1,0)
        oovv = eris.oovv.copy()
        vvvv = eris.vvvv.copy()
        ovvo = eris.ovvo.copy()

        eijabc = eris.get_eijabc(**kwargs)

        e = []
        for _eval, _evec, _levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
            l1,l2 = self.vector_to_amplitudes_ea(_levec, **kwargs)
            r1,r2 = self.vector_to_amplitudes_ea(_evec, **kwargs)
            ldotr = dot(l1.conj(),r1) + dot(l2.ravel(),r2.ravel())
            l1 /= ldotr
            l2 /= ldotr
            l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
            r2 = r2.transpose(0,2,1)

            _eijabc = eijabc + _eval
            _eijabc = 1./_eijabc

            lijabc = -0.5*einsum('c,iajb->ijabc', l1, ovov)
            lijabc += einsum('jmia,mbc->ijabc', ooov, l2)
            lijabc -= einsum('iaeb,jec->ijabc', ovvv, l2)
            lijabc -= einsum('jbec,iae->ijabc', ovvv, l2)
            lijabc = lijabc + lijabc.transpose(1,0,3,2,4)

            rijabc = -einsum('becf,f,ijae->ijabc', vvvv, r1, t2)
            rijabc += einsum('mjce,e,imab->ijabc', oovv, r1, t2)
            rijabc += einsum('mebj,e,imac->ijabc', ovvo, r1, t2)
            rijabc += einsum('aimj,mbc->ijabc', vooo, r2)
            rijabc += -einsum('bjce,iae->ijabc', vovv, r2)
            rijabc += -einsum('aibe,jec->ijabc', vovv, r2)
            rijabc = rijabc + rijabc.transpose(1,0,3,2,4)

            lijabc =  4.*lijabc \
                    - 2.*lijabc.transpose(0,1,3,2,4) \
                    - 2.*lijabc.transpose(0,1,4,3,2) \
                    - 2.*lijabc.transpose(0,1,2,4,3) \
                    + 1.*lijabc.transpose(0,1,3,4,2) \
                    + 1.*lijabc.transpose(0,1,4,2,3)
            deltaE = 0.5*einsum('ijabc,ijabc',lijabc,rijabc*_eijabc)
            deltaE = deltaE.real
            logger.note(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        _eval+deltaE, deltaE)
            e.append(_eval+deltaE)
        return e


    def eeccsd(self, nroots=1, koopmans=False, guess=None, partition=None):
        '''Calculate N-electron neutral excitations via EE-EOM-CCSD.

        Kwargs:
            nroots : int
                Number of roots (eigenvalues) requested
            partition : bool or str
                Use a matrix-partitioning for the doubles-doubles block.
                Can be None, 'mp' (Moller-Plesset, i.e. orbital energies on the diagonal),
                or 'full' (full diagonal elements).
            koopmans : bool
                Calculate Koopmans'-like (1p1h) excitations only, targeting via
                overlap.
            guess : list of ndarray
                List of guess vectors to use for targeting via overlap.
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nee()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ee_partition = partition
        if partition == 'full':
            self._eeccsd_diag_matrix2 = self.vector_to_amplitudes_ee(self.eeccsd_diag())[1]

        nvir = self.nmo - self.nocc
        adiag = self.eeccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            guess = []
            idx = adiag.argsort()
            n = 0
            for i in idx:
                g = np.zeros(size)
                g[i] = 1.0
                if koopmans:
                    if np.linalg.norm(g[:self.nocc*nvir])**2 > 0.8:
                        guess.append(g)
                        n += 1
                else:
                    guess.append(g)
                    n += 1
                if n == nroots:
                    break

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        eig = linalg_helper.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = linalg_helper._gen_x0(envs['v'], envs['xs'])
                idx = np.argmax( np.abs(np.dot(np.array(guess).conj(),np.array(x0).T)), axis=1 )
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx)
            eee, evecs = eig(self.eeccsd_matvec, guess, precond, pick=pickeig,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
        else:
            eee, evecs = eig(self.eeccsd_matvec, guess, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eee = eee.real

        if nroots == 1:
            eee, evecs = [self.eee], [evecs]
        for n, en, vn in zip(range(nroots), eee, evecs):
            logger.info(self, 'EE root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:self.nocc*nvir])**2)
        log.timer('EE-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs

    def eeccsd_matvec(self,vector,**kwargs):
        raise NotImplementedError

    def eeccsd_diag(self):
        raise NotImplementedError

    def ipccsd_t_star(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_t3p2_ip_imds:
            self.imds.make_t3p2_ip(self)
        return self.ipccsd(nroots, left, koopmans, guess, partition)

    def eaccsd_t_star(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_t3p2_ea_imds:
            self.imds.make_t3p2_ea(self)
        return self.eaccsd(nroots, left, koopmans, guess, partition)

    def eeccsd_matvec_singlet(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds
        r1, r2 = self.vector_to_amplitudes_singlet(vector, **kwargs)
        # rbar_ijab = 2 r_ijab - r_ijba
        r2bar = 2.*r2 - r2.transpose(0,1,3,2)
        # wbar_nmie = 2 W_nmie - W_nmei = 2 W_nmie - W_mnie
        woOoV_bar = 2. * imds.woOoV - imds.woOoV.transpose(1,0,2,3)
        # wbar_amfe = 2 W_amfe - W_amef
        wvOvV_bar = 2. * imds.wvOvV - imds.wvOvV.transpose(0,1,3,2)
        # wbar_mbej = 2 W_mbej - W_mbje
        woVvO_bar = 2. * imds.woVvO - imds.woVoV.transpose(0,1,3,2)

        Hr1 = -einsum('mi,ma->ia', imds.Foo, r1)
        # r_ia <- F_ac r_ic
        Hr1 += einsum('ac,ic->ia', imds.Fvv, r1)

        Hr1 += 2. * einsum('maei,me->ia', imds.woVvO, r1)
        Hr1 -= einsum('maie,me->ia', imds.woVoV, r1)

            # r_ia <- F_me (2 r_imae - r_miae)
        Hr1 += 2. * einsum('me,imae->ia', imds.Fov, r2)
        Hr1 -= einsum('me,miae->ia', imds.Fov, r2)

        # r_ia <- (2 W_amef - W_amfe) r_imef
        Hr1 += 2. * einsum('amef,imef->ia', imds.wvOvV, r2)
        Hr1 -= einsum('amfe,imef->ia', imds.wvOvV, r2)

        # r_ia <- -W_mnie (2 r_mnae - r_nmae)
        Hr1 -= 2. * einsum('mnie,mnae->ia', imds.woOoV, r2)
        Hr1 += einsum('mnie,nmae->ia', imds.woOoV, r2)

        # r_ijab <= - F_mj r_imab
        Hr2 -= einsum('mj,imab->ijab', imds.Foo, r2)
        # r_ijab <= - F_mi r_jmba
        Hr2 -= einsum('mi,jmba->ijab', imds.Foo, r2)
        # r_ijab <= F_be r_ijae
        Hr2 += einsum('be,ijae->ijab', imds.Fvv, r2)
        # r_ijab <= F_ae r_jibe
        Hr2 += einsum('ae,jibe->ijab', imds.Fvv, r2)

        # r_ijab <= W_abej r_ie
        Hr2 += einsum('abej,ie->ijab', imds.wvVvO, r1)
        # r_ijab <= W_baei r_je
        Hr2 += einsum('baei,je->ijab', imds.wvVvO, r1)
        # r_ijab <= - W_mbij r_ma
        Hr2 -= einsum('mbij,ma->ijab', imds.woVoO, r1)
        # r_ijab <= - W_maji r_mb
        Hr2 -= einsum('maji,mb->ijab', imds.woVoO, r1)

        # r_ijab <= (2 W_mbej - W_mbje) r_imae - W_mbej r_imea
        tmp  = einsum('mbej,imae->ijab', woVvO_bar, r2)
        tmp -= einsum('mbej,imea->ijab', imds.woVvO, r2)
        # r_ijab <= - W_maje r_imeb
        tmp -= einsum('maje,imeb->ijab', imds.woVoV, r2)
        Hr2 += tmp
        # The following two lines can be obtained by simply transposing tmp:
        #   r_ijab <= (2 W_maei - W_maie) r_jmbe - W_maei r_jmeb
        #   r_ijab <= - W_mbie r_jmea
        Hr2 += tmp.transpose(1,0,3,2)
        tmp = None

        Hr2 += einsum('abef,ijef->ijab', imds.wvVvV, r2)
        # r_ijab <= W_mnij r_mnab
        Hr2 += einsum('mnij,mnab->ijab', imds.woOoO, r2)

        # Wr2_jm = W_mnef (2 r_jnef - r_jnfe) = W_mnef rbar_jnef
        wr2_oo = einsum('mnef,jnef->jm', imds.woOvV, r2bar)
        # Wr2_eb = W_mnef (2 r_mnbf - r_mnfb) = W_mnef rbar_mnbf
        wr2_vv = einsum('mnef,mnbf->eb', imds.woOvV, r2bar)
        # Wr1_in = (2 W_nmie - W_nmei) r_me = wbar_nmie r_me
        wr1_oo = einsum('nmie,me->in', woOoV_bar, r1)
        # Wr1_fa = (2 W_amfe - W_amef) r_me = wbar_amfe r_me
        wr1_vv = einsum('amfe,me->fa', wvOvV_bar, r1)

        # r_ijab <= - Wr2_jm t_imab
        Hr2 -= einsum('jm,imab->ijab', wr2_oo, imds.t2)
        # r_ijab <= - Wr2_im t_jmba
        Hr2 -= einsum('im,jmba->ijab', wr2_oo, imds.t2)
        # r_ijab <= - Wr2_eb t_ijae
        Hr2 -= einsum('eb,ijae->ijab', wr2_vv, imds.t2)
        # r_ijab <= - Wr2_ea t_jibe
        Hr2 -= einsum('ea,jibe->ijab', wr2_vv, imds.t2)

        # r_ijab <= - Wr1_in t_jnba
        Hr2 -= einsum('in,jnba->ijab', wr1_oo, imds.t2)
        # r_ijab <= - Wr1_jn t_inab
        Hr2 -= einsum('jn,inab->ijab', wr1_oo, imds.t2)
        # r_ijab <= Wr1_fa t_jibf
        Hr2 += einsum('fa,jibf->ijab', wr1_vv, imds.t2)
        # r_ijab <= Wr1_fb t_ijaf
        Hr2 += einsum('fb,ijaf->ijab', wr1_vv, imds.t2)

        vector = self.amplitudes_to_vector_singlet(Hr1,Hr2,**kwargs)
        return vector

class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))
        nocc = cc.nocc
        nmo = cc.nmo
        mo_e = fock.diagonal().real
        self.eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        self.eijab = lib.direct_sum('ia,jb->ijab',self.eia, self.eia)
        self.foo = fock[:nocc,:nocc]
        self.fov = fock[:nocc,nocc:]
        self.fvv = fock[nocc:,nocc:]
        self._foo = np.diag(np.diag(self.foo))
        self._fvv = np.diag(np.diag(self.fvv))
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
        self.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()

    def get_eijkab(self, **kwargs):
        '''compute ei+ej+ek-ea-eb'''
        nocc, nvir = self.eia.shape
        eijkab = self.eijab.reshape(nocc,nocc,1,nvir,nvir) + self.foo.diagonal().reshape(1,1,nocc,1,1)
        return eijkab

    def get_eijabc(self, **kwargs):
        '''compute ei+ej+ek-ea-eb'''
        nocc, nvir = self.eia.shape
        eijabc = self.eijab.reshape(nocc,nocc,nvir,nvir,1) - self.fvv.diagonal().reshape(1,1,1,1,nvir)
        return eijabc

    def get_eijkabc(self, **kwargs):
        '''compute ei+ej+ek-ea-eb'''
        nocc, nvir = self.eia.shape
        eijabc = self.eijab.reshape(nocc,nocc,1,nvir,nvir,1) + self.eia.reshape(1,1,nocc,1,1,nvir)
        return eijabc

class _IMDS:
    def __init__(self, cc):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if getattr(cc, 'eris', None) is not None:
            self.eris = cc.eris
        else:
            self.eris = cc.ao2mo()
        self.made_ip_imds = False
        self.made_ea_imds = False
        self._made_shared_2e = False

        self.made_t3p2_ip_imds = False
        self.made_t3p2_ea_imds = False

    def _make_shared_1e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1,t2,eris)
        self.Lvv = imd.Lvv(t1,t2,eris)
        self.Fov = imd.cc_Fov(t1,t2,eris)

        log.timer('EOM-CCSD shared one-electron intermediates', *cput0)

    def _make_shared_2e(self):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov = imd.Wovov(t1,t2,eris)
        self.Wovvo = imd.Wovvo(t1,t2,eris)
        self.Woovv = eris.ovov.transpose(0,2,1,3)

        log.timer('EOM-CCSD shared two-electron intermediates', *cput0)

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ip_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo = imd.Woooo(t1,t2,eris)
        self.Wooov = imd.Wooov(t1,t2,eris)
        self.Wovoo = imd.Wovoo(t1,t2,eris)
        self.made_ip_imds = True
        log.timer('EOM-CCSD IP intermediates', *cput0)

    def make_t3p2_ip(self, cc):
        assert(cc.ip_partition is None)
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo
        self.made_t3p2_ip_imds = True

        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
        return self

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if self._made_shared_2e is False and ea_partition != 'mp':
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1,t2,eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1,t2,eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1,t2,eris)
        else:
            self.Wvvvv = imd.Wvvvv(t1,t2,eris)
            self.Wvvvo = imd.Wvvvo(t1,t2,eris,self.Wvvvv)
        self.made_ea_imds = True
        log.timer('EOM-CCSD EA intermediates', *cput0)

    def make_t3p2_ea(self, cc, ea_partition=None):
        assert(ea_partition is None)
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared_2e = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo
        self.made_t3p2_ea_imds = True

        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
        return self

    def make_ee(self):
        self._make_shared_1e()
        if self._made_shared_2e is False:
            self._make_shared_2e()
            self._made_shared_2e = True

        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # Rename imds to match the notations in pyscf.cc.eom_rccsd
        self.Foo = self.Loo
        self.Fvv = self.Lvv
        self.woOvV = self.Woovv
        self.woVvO = self.Wovvo
        self.woVoV = self.Wovov

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.woOoO = imd.Woooo(t1, t2, eris)
            self.woOoV = imd.Wooov(t1, t2, eris)
            self.woVoO = imd.Wovoo(t1, t2, eris)
        else:
            self.woOoO = self.Woooo
            self.woOoV = self.Wooov
            self.woVoO = self.Wovoo
            if self.made_t3p2_ip_imds:
                logger.warn(self, 'imds.t1/t2/Wovoo is detected to carry t3p2 contribution. '
                            'This could be due to previous execution of ipccsd_t_star.\n'
                            'Please make sure this is the expected behavior, otherwise please reset imds')

        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.wvOvV = imd.Wvovv(t1, t2, eris)
            self.wvVvV = imd.Wvvvv(t1, t2, eris)
            self.wvVvO = imd.Wvvvo(t1, t2, eris, self.wvVvV)
        else:
            self.wvOvV = self.Wvovv
            self.wvVvV = self.Wvvvv
            self.wvVvO = self.Wvvvo
            if self.made_t3p2_ea_imds:
                logger.warn(self, 'imds.t1/t2/Wvvvo is detected to carry t3p2 contribution. '
                            'This could be due to previous execution of eaccsd_t_star.\n'
                            'Please make sure this is the expected behavior, otherwise please reset imds')

        self.made_ee_imds = True
        log.timer('EOM-CCSD EE intermediates', *cput0)

if __name__ == '__main__':
    from pyscf import scf, gto

    mol = gto.M()
    nocc, nvir = 5, 12
    nmo = nocc + nvir
    nmo_pair = nmo*(nmo+1)//2
    mf = scf.RHF(mol)
    np.random.seed(12)
    mf._eri = np.random.random(nmo_pair*(nmo_pair+1)//2)
    mf.mo_coeff = np.random.random((nmo,nmo))
    mf.mo_energy = np.arange(0., nmo)
    mf.mo_occ = np.zeros(nmo)
    mf.mo_occ[:nocc] = 2
    vhf = mf.get_veff(mol, mf.make_rdm1())
    cinv = np.linalg.inv(mf.mo_coeff)
    mf.get_hcore = lambda *args: (reduce(np.dot, (cinv.T*mf.mo_energy, cinv)) - vhf)
    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    a = np.random.random((nmo,nmo)) * .1
    a+= a.T.conj()
    eris.foo = eris.foo + a[:nocc,:nocc]
    eris.fov = eris.fov + a[:nocc,nocc:]
    eris.fvv = eris.fvv + a[nocc:,nocc:]
    eris._foo = np.diag(np.diag(eris.foo))
    eris._fvv = np.diag(np.diag(eris.fvv))
    eris.eia = eris.foo.diagonal()[:,None] - eris.fvv.diagonal()[None,:]
    eris.eijab = eris.eia[:,None,:,None] + eris.eia[None,:,None,:]
    eris.eia = eris.eia.real
    eris.eijab = eris.eijab.real

    t1 = np.random.random((nocc,nvir)) * .1
    t2 = np.random.random((nocc,nocc,nvir,nvir)) * .1
    t2 = t2 + t2.transpose(1,0,3,2)
    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - 66540.100267798145)
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - -106360.5276951083)
    print(lib.finger(t2a) - -1517.9391800662809)
    eri1 = np.random.random((nmo,nmo,nmo,nmo)) + np.random.random((nmo,nmo,nmo,nmo))*1j
    eri1 = eri1.transpose(0,2,1,3)
    eri1 = eri1 + eri1.transpose(1,0,3,2).conj()
    eri1 = eri1 + eri1.transpose(2,3,0,1)
    eri1 *= .1
    eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
    eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
    eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovvo = eri1[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri1[nocc:,nocc:,nocc:,nocc:].copy()
    a = np.random.random((nmo,nmo)) * .1j
    a += a.T.conj()

    eris.foo = eris.foo + a[:nocc,:nocc]
    eris.fov = eris.fov + a[:nocc,nocc:]
    eris.fvv = eris.fvv + a[nocc:,nocc:]
    eris._foo = np.diag(np.diag(eris.foo))
    eris._fvv = np.diag(np.diag(eris.fvv))
    eris.eia = eris.foo.diagonal()[:,None] - eris.fvv.diagonal()[None,:]
    eris.eijab = eris.eia[:,None,:,None] + eris.eia[None,:,None,:]
    eris.eia = eris.eia.real
    eris.eijab = eris.eijab.real

    t1 = t1 + np.random.random((nocc,nvir)) * .1j
    t2 = t2 + np.random.random((nocc,nocc,nvir,nvir)) * .1j
    t2 = t2 + t2.transpose(1,0,3,2)
    mycc.cc2 = False
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (9.2521062044785189+29.999480274811873j))
    mycc.cc2 = True
    t1a, t2a = update_amps(mycc, t1, t2, eris)
    print(lib.finger(t1a) - (-13.32050019680894-1.8825765910430254j))
    print(lib.finger(t2a) - (-0.056223856104895858+0.025472249329733986j))

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    #mol.basis = '3-21G'
    mol.verbose = 0
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    emp2, t1, t2 = mycc.init_amps(eris)
    print(lib.finger(t2) - -0.03928989832225917)
    np.random.seed(1)
    t1 = np.random.random(t1.shape)*.1
    t2 = np.random.random(t2.shape)*.1
    t2 = t2 + t2.transpose(1,0,3,2)
    t1, t2 = update_amps(mycc, t1, t2, eris)

    print(lib.finger(t1) - 0.275281632153816)
    print(lib.finger(t2) - 0.17686480293354967)

    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.21334326214236796)

    ecct = mycc.ccsd_t()
    print(ecct - -0.003060023300574146)

    ecct = mycc.ccsd_t_slow()
    print(ecct - -0.003060023300574146)

    print("IP energies... (right eigenvector)")
    e,v = mycc.ipccsd(nroots=3)
    print(e[0] - 0.43356041409195489)
    print(e[1] - 0.51876598058509493)
    print(e[2] - 0.6782879569941862 )

    print("IP energies... (left eigenvector)")
    le,lv = mycc.ipccsd(nroots=3,left=True)
    print(le[0] - 0.43356040428879794)
    print(le[1] - 0.51876597800180335)
    print(le[2] - 0.67828755013874864)

    e = mycc.ipccsd_star_contract(e,v,lv)
    print(e[0] - 0.43793202073189047)
    print(e[1] - 0.52287073446559729)
    print(e[2] - 0.67994597948852287)

    print("EA energies... (right eigenvector)")
    e,v = mycc.eaccsd(nroots=3)
    print(e[0] - 0.16737886282063008)
    print(e[1] - 0.24027622989542635)
    print(e[2] - 0.51006796667905585)

    print("EA energies... (left eigenvector)")
    le,lv = mycc.eaccsd(nroots=3,left=True)
    print(le[0] - 0.16737896537079733)
    print(le[1] - 0.24027634198123343)
    print(le[2] - 0.51006809015066612)

    e = mycc.eaccsd_star_contract(e,v,lv)
    print(e[0] - 0.16656250989594673)
    print(e[1] - 0.23944144399509584)
    print(e[2] - 0.41399436420039226)

    eip, v= mycc.ipccsd_t_star(nroots=3)

    print(eip[0] - 0.43455702)
    print(eip[1] - 0.51991415)
    print(eip[2] - 0.67944536)

    eea, v= mycc.eaccsd_t_star(nroots=3)

    print(eea[0] - 0.16785694)
    print(eea[1] - 0.24098345)
    print(eea[2] - 0.51127641)
