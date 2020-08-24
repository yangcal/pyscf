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

import time
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import gccsd, eom_gccsd, rccsd_slow
from pyscf.cc import gintermediates_slow as imd

einsum = lib.einsum
asarray = np.asarray
dot = np.dot

# GCCSD with antisymmetrized integrals

def update_amps(mycc, t1, t2, eris):
    tau = imd.make_tau(t2, t1, t1, eris)
    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    Woooo = imd.cc_Woooo(t1, t2, eris)
    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    Wovvo = imd.cc_Wovvo(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= eris._fvv
    Foo -= eris._foo

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += eris.fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += eris.oovv.conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', tau, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    tmp1 = einsum('ma,mbje->abje', t1, eris.ovov)
    tmp += einsum('ie,abje->ijab', t1, tmp1)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, eris.ovvv.conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, eris.ooov.conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))
    t1new /= eris.eia
    t2new /= eris.eijab
    return t1new, t2new

def make_intermediates(mycc, t1, t2, eris):
    foo = eris.foo
    fov = eris.fov
    fvo = eris.fov.conj().transpose(1,0)
    fvv = eris.fvv

    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2

    v1 = fvv - einsum('ja,jb->ba', fov, t1)
    v1-= einsum('jbac,jc->ba', eris.ovvv, t1)
    v1+= einsum('jkca,jkbc->ba', eris.oovv, tau) * .5

    v2 = foo + einsum('ib,jb->ij', fov, t1)
    v2-= einsum('kijb,kb->ij', eris.ooov, t1)
    v2+= einsum('ikbc,jkbc->ij', eris.oovv, tau) * .5

    v3 = einsum('ijcd,klcd->ijkl', eris.oovv, tau)
    v4 = einsum('ljdb,klcd->jcbk', eris.oovv, t2)
    v4+= eris.ovvo

    v5 = fvo + einsum('kc,jkbc->bj', fov, t2)
    tmp = fov - einsum('kldc,ld->kc', eris.oovv, t1)
    v5+= einsum('kc,kb,jc->bj', tmp, t1, t1)
    v5-= einsum('kljc,klbc->bj', eris.ooov, t2) * .5
    v5+= einsum('kbdc,jkcd->bj', eris.ovvv, t2) * .5

    w3 = v5 + einsum('jcbk,jb->ck', v4, t1)
    w3 += einsum('cb,jb->cj', v1, t1)
    w3 -= einsum('jk,jb->bk', v2, t1)

    woooo = eris.oooo * .5
    woooo+= v3 * .25
    woooo+= einsum('jilc,kc->jilk', eris.ooov, t1)

    wovvo = v4 - einsum('ljdb,lc,kd->jcbk', eris.oovv, t1, t1)
    wovvo-= einsum('ljkb,lc->jcbk', eris.ooov, t1)
    wovvo+= einsum('jcbd,kd->jcbk', eris.ovvv, t1)

    wovoo = einsum('icdb,jkdb->icjk', eris.ovvv, tau) * .25
    wovoo+= einsum('jkic->icjk', eris.ooov.conj()) * .5
    wovoo+= einsum('icbk,jb->icjk', v4, t1)
    wovoo-= einsum('lijb,klcb->icjk', eris.ooov, t2)

    wvvvo = einsum('jcak,jb->bcak', v4, t1)
    wvvvo+= einsum('jlka,jlbc->bcak', eris.ooov, tau) * .25
    wvvvo-= einsum('jacb->bcaj', eris.ovvv.conj()) * .5
    wvvvo+= einsum('kbad,jkcd->bcaj', eris.ovvv, t2)

    class _LIMDS: pass
    imds = _LIMDS()
    imds.woooo = woooo
    imds.wovvo = wovvo
    imds.wovoo = wovoo
    imds.wvvvo = wvvvo
    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    fov = eris.fov
    v1 = imds.v1 - eris._fvv
    v2 = imds.v2 - eris._foo

    mba = einsum('klca,klcb->ba', l2, t2) * .5
    mij = einsum('kicd,kjcd->ij', l2, t2) * .5
    m3 = einsum('klab,ijkl->ijab', l2, imds.woooo)
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    tmp = einsum('ijcd,klcd->ijkl', l2, tau)
    oovv = eris.oovv
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = einsum('ijcd,kd->ijck', l2, t1)
    m3 -= einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    m3 += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

    l2new = oovv + m3
    fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
    tmp = einsum('ia,jb->ijab', l1, fov1)
    tmp+= einsum('kica,jcbk->ijab', l2, imds.wovvo)
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,ijkb->ijab', l1, eris.ooov)
    tmp+= einsum('ijca,cb->ijab', l2, v1)
    tmp1vv = mba + einsum('ka,kb->ba', l1, t1)
    tmp+= einsum('ca,ijcb->ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ic,jcba->jiba', l1, eris.ovvv)
    tmp+= einsum('kiab,jk->ijab', l2, v2)
    tmp1oo = mij + einsum('ic,kc->ik', l1, t1)
    tmp-= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)

    l1new = fov + einsum('jb,ibaj->ia', l1, eris.ovvo)
    l1new += einsum('ib,ba->ia', l1, v1)
    l1new -= einsum('ja,ij->ia', l1, v2)
    l1new -= einsum('kjca,icjk->ia', l2, imds.wovoo)
    l1new -= einsum('ikbc,bcak->ia', l2, imds.wvvvo)
    l1new += einsum('ijab,jb->ia', m3, t1)
    l1new += einsum('jiba,bj->ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('bd,jd->jb', tmp1vv, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += einsum('jiba,jb->ia', oovv, tmp)
    l1new += einsum('icab,bc->ia', eris.ovvv, tmp1vv)
    l1new -= einsum('jika,kj->ia', eris.ooov, tmp1oo)
    tmp = fov - einsum('kjba,jb->ka', oovv, t1)
    l1new -= einsum('ik,ka->ia', mij, tmp)
    l1new -= einsum('ca,ic->ia', mba, tmp)

    l1new /= eris.eia
    l2new /= eris.eijab

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

def energy(mycc, t1, t2, eris, fac=1.0):
    e = einsum('ia,ia', eris.fov, t1)
    e += 0.25*einsum('ijab,ijab', t2, eris.oovv)
    tmp = einsum('ia,ijab->jb', t1, eris.oovv)
    e += 0.5*einsum('jb,jb->', t1, tmp)
    e*= fac
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in GCCSD energy %s', e)
    return e.real

class GCCSD(gccsd.GCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self.ip_partition = self.ea_partition = None
        self._keys = self._keys.union(['max_space', 'ip_partition', 'ea_partition'])

    energy = energy
    update_amps = update_amps

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        t1 = eris.fov / eris.eia
        t2 = eris.oovv / eris.eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris.oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def ao2mo(self, mo_coeff=None):
        return _make_eris_incore(self, mo_coeff)

    nip = eom_gccsd.EOMIP.vector_size
    nea = eom_gccsd.EOMEA.vector_size
    nee = eom_gccsd.EOMEE.vector_size

    amplitudes_to_vector_ip = eom_gccsd.EOMIP.amplitudes_to_vector
    amplitudes_to_vector_ea = eom_gccsd.EOMEA.amplitudes_to_vector
    amplitudes_to_vector_ee = eom_gccsd.EOMEE.amplitudes_to_vector

    def vector_to_amplitudes_ip(self, vector, **kwargs):
        return eom_gccsd.EOMIP.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_ea(self, vector, **kwargs):
        return eom_gccsd.EOMEA.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_ee(self, vector, **kwargs):
        return eom_gccsd.EOMEE.vector_to_amplitudes(self, vector)

    def ipccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()

        imds = self.imds
        t1, t2 = imds.t1, imds.t2
        nocc, nvir = t1.shape

        Hr1 = -imds.Foo.diagonal()
        Hr2 = imds.Fvv.diagonal().reshape(1,1,nvir) -\
              imds.Foo.diagonal().reshape(nocc,1,1) -\
              imds.Foo.diagonal().reshape(1,nocc,1)

        Hr2 = Hr2 + .5 * einsum('ijij->ij', imds.Woooo).reshape(nocc,nocc,1) -\
              .5 * einsum('jiij->ij',imds.Woooo).reshape(nocc,nocc,1)

        Hr2 = Hr2 + einsum('iaai->ia', imds.Wovvo).reshape(nocc,1,nvir) +\
              einsum('jaaj->ja', imds.Wovvo).reshape(1,nocc,nvir)
        Hr2+= .5 * einsum('ijea,ijae->ija', imds.Woovv, t2) - \
              .5 * einsum('jiea,ijae->ija', imds.Woovv, t2)
        vector = self.amplitudes_to_vector_ip(Hr1, Hr2)
        return vector

    def eaccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds
        t2 = imds.t2
        nocc, nvir = imds.t1.shape
        Hr1 = imds.Fvv.diagonal()
        Hr2 = imds.Fvv.diagonal().reshape(1,nvir,1) +\
              imds.Fvv.diagonal().reshape(1,1,nvir) -\
              imds.Foo.diagonal().reshape(nocc,1,1)
        Hr2 = Hr2 + einsum('jbbj->jb', imds.Wovvo).reshape(nocc,1,nvir) +\
              einsum('jaaj->ja', imds.Wovvo).reshape(nocc,nvir,1)

        Hr2 = Hr2 + .5 * einsum('abab->ab', imds.Wvvvv).reshape(1,nvir,nvir) -\
              .5 * einsum('abba->ab', imds.Wvvvv).reshape(1,nvir,nvir)
        Hr2 -= .5 * einsum('ijab,ijab->jab', imds.Woovv, t2) -\
               .5 * einsum('ijba,ijab->jab', imds.Woovv, t2)

        vector = self.amplitudes_to_vector_ea(Hr1, Hr2)
        return vector


    def eeccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds
        t2 = imds.t2
        nocc, nvir = imds.t1.shape
        Hr1 = imds.Fvv.diagonal().reshape(1,nvir) - imds.Foo.diagonal().reshape(nocc,1) +\
              einsum('iaai->ia', imds.Wovvo)

        tmp = .5 *(einsum('ijeb,ijbe->ijb', imds.Woovv, t2) -\
                   einsum('jieb,ijbe->ijb', imds.Woovv, t2))
        Hr2 = imds.Fvv.diagonal().reshape(1,1,1,nvir) + tmp.reshape(nocc,nocc,nvir,1)
        Hr2 += (imds.Fvv.diagonal().reshape(1,1,nvir,1) + tmp.reshape(nocc,nocc,1,nvir))
        Hr2 = Hr2 + .5 * einsum('abab->ab', imds.Wvvvv).reshape(1,1,nvir,nvir) -\
              .5 * einsum('abba->ab', imds.Wvvvv).reshape(1,1,nvir,nvir)
        tmp = einsum('iaai->ia', imds.Wovvo)

        Hr2 += (tmp.reshape(nocc,1,nvir,1) + tmp.reshape(nocc,1,1,nvir) +\
                tmp.reshape(1,nocc,nvir,1) + tmp.reshape(1,nocc,1,nvir))

        tmp = .5*(einsum('kjab,jkab->jab', imds.Woovv, t2) -\
                  einsum('kjba,jkab->jab', imds.Woovv, t2))
        Hr2 += (-imds.Foo.diagonal().reshape(1,nocc,1,1) + tmp.reshape(nocc,1,nvir,nvir))
        Hr2 += (-imds.Foo.diagonal().reshape(nocc,1,1,1) + tmp.reshape(1,nocc,nvir,nvir))
        Hr2 = Hr2 + .5*einsum('ijij->ij', imds.Woooo).reshape(nocc,nocc,1,1) -\
                    .5*einsum('jiij->ij', imds.Woooo).reshape(nocc,nocc,1,1)
        return self.amplitudes_to_vector_ee(Hr1, Hr2)

    def ipccsd_matvec(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        r1, r2 = self.vector_to_amplitudes_ip(vector, **kwargs)
        # Eq. (8)
        Hr1 = -einsum('mi,m->i', imds.Foo, r1)
        Hr1 += einsum('me,mie->i', imds.Fov, r2)
        Hr1 += -0.5*einsum('nmie,mne->i', imds.Wooov, r2)
        # Eq. (9)
        Hr2 =  einsum('ae,ije->ija', imds.Fvv, r2)
        tmp1 = einsum('mi,mja->ija', imds.Foo, r2)
        Hr2 -= tmp1 - tmp1.transpose(1,0,2)
        Hr2 -= einsum('maji,m->ija', imds.Wovoo, r1)
        Hr2 += 0.5*einsum('mnij,mna->ija', imds.Woooo, r2)
        tmp2 = einsum('maei,mje->ija', imds.Wovvo, r2)
        Hr2 += tmp2 - tmp2.transpose(1,0,2)
        Hr2 += 0.5*einsum('mnef,mnf,ijae->ija', imds.Woovv, r2, imds.t2)
        vector = self.amplitudes_to_vector_ip(Hr1, Hr2)
        return vector

    def lipccsd_matvec(self, vector, **kwargs):
        '''IP-CCSD left eigenvector equation.

        For description of args, see ipccsd_matvec.'''
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        r1, r2 = self.vector_to_amplitudes_ip(vector, **kwargs)

        Hr1 = -einsum('mi,i->m', imds.Foo, r1)
        Hr1 += -0.5 * einsum('maji,ija->m', imds.Wovoo, r2)

        Hr2 = einsum('me,i->mie', imds.Fov, r1)
        Hr2 -= einsum('ie,m->mie', imds.Fov, r1)
        Hr2 += -einsum('nmie,i->mne', imds.Wooov, r1)
        Hr2 += einsum('ae,ija->ije', imds.Fvv, r2)
        tmp1 = einsum('mi,ija->mja', imds.Foo, r2)
        Hr2 += (-tmp1 + tmp1.transpose(1, 0, 2))
        Hr2 += 0.5 * einsum('mnij,ija->mna', imds.Woooo, r2)
        tmp2 = einsum('maei,ija->mje', imds.Wovvo, r2)
        Hr2 += (tmp2 - tmp2.transpose(1, 0, 2))
        Hr2 += 0.5 * einsum('mnef,ija,ijae->mnf', imds.Woovv, r2, imds.t2)

        vector = self.amplitudes_to_vector_ip(Hr1, Hr2)
        return vector

    def eaccsd_matvec(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes_ea(vector, **kwargs)

        # Eq. (30)
        Hr1  = einsum('ac,c->a', imds.Fvv, r1)
        Hr1 += einsum('ld,lad->a', imds.Fov, r2)
        Hr1 += 0.5*einsum('alcd,lcd->a', imds.Wvovv, r2)
        # Eq. (31)
        Hr2 = einsum('abcj,c->jab', imds.Wvvvo, r1)
        tmp1 = einsum('ac,jcb->jab', imds.Fvv, r2)
        Hr2 += tmp1 - tmp1.transpose(0,2,1)
        Hr2 -= einsum('lj,lab->jab', imds.Foo, r2)
        tmp2 = einsum('lbdj,lad->jab', imds.Wovvo, r2)
        Hr2 += tmp2 - tmp2.transpose(0,2,1)
        Hr2 += 0.5*einsum('abcd,jcd->jab', imds.Wvvvv, r2)
        Hr2 -= 0.5*einsum('klcd,lcd,kjab->jab', imds.Woovv, r2, imds.t2)
        vector = self.amplitudes_to_vector_ea(Hr1, Hr2)
        return vector

    def leaccsd_matvec(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes_ea(vector, **kwargs)

        # Eq. (32)
        Hr1 = einsum('ac,a->c',imds.Fvv,r1)
        Hr1 += 0.5*einsum('abcj,jab->c',imds.Wvvvo,r2)
        # Eq. (33)
        Hr2 = einsum('alcd,a->lcd',imds.Wvovv,r1)
        Hr2 += einsum('ld,a->lad',imds.Fov,r1)
        Hr2 -= einsum('la,d->lad',imds.Fov,r1)
        tmp1 = einsum('ac,jab->jcb',imds.Fvv,r2)
        Hr2 += (tmp1 - tmp1.transpose(0,2,1))
        Hr2 += -einsum('lj,jab->lab',imds.Foo,r2)
        tmp2 = einsum('lbdj,jab->lad',imds.Wovvo,r2)
        Hr2 += (tmp2 - tmp2.transpose(0,2,1))
        Hr2 += 0.5*einsum('abcd,jab->jcd',imds.Wvvvv,r2)
        Hr2 += -0.5*einsum('klcd,jab,kjab->lcd',imds.Woovv,r2,imds.t2)

        vector = self.amplitudes_to_vector_ea(Hr1,Hr2)
        return vector

    def eeccsd_matvec(self, vector, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ee_imds:
            self.imds.make_ee()
        imds = self.imds
        r1, r2 = self.vector_to_amplitudes_ee(vector, **kwargs)

        # Eq. (9)
        Hr1  = einsum('ae,ie->ia', imds.Fvv, r1)
        Hr1 -= einsum('mi,ma->ia', imds.Foo, r1)
        Hr1 += einsum('me,imae->ia', imds.Fov, r2)
        Hr1 += einsum('maei,me->ia', imds.Wovvo, r1)
        Hr1 -= 0.5*einsum('mnie,mnae->ia', imds.Wooov, r2)
        Hr1 += 0.5*einsum('amef,imef->ia', imds.Wvovv, r2)
        # Eq. (10)
        tmpab = einsum('be,ijae->ijab', imds.Fvv, r2)
        tmpab -= 0.5*einsum('mnef,ijae,mnbf->ijab', imds.Woovv, imds.t2, r2)
        tmpab -= einsum('mbij,ma->ijab', imds.Wovoo, r1)
        tmpab -= einsum('amef,ijfb,me->ijab', imds.Wvovv, imds.t2, r1)
        tmpij  = einsum('mj,imab->ijab', -imds.Foo, r2)
        tmpij -= 0.5*einsum('mnef,imab,jnef->ijab', imds.Woovv, imds.t2, r2)
        tmpij += einsum('abej,ie->ijab', imds.Wvvvo, r1)
        tmpij += einsum('mnie,njab,me->ijab', imds.Wooov, imds.t2, r1)

        tmpabij = einsum('mbej,imae->ijab', imds.Wovvo, r2)
        tmpabij = tmpabij - tmpabij.transpose(1,0,2,3)
        tmpabij = tmpabij - tmpabij.transpose(0,1,3,2)
        Hr2 = tmpabij

        Hr2 += tmpab - tmpab.transpose(0,1,3,2)
        Hr2 += tmpij - tmpij.transpose(1,0,2,3)
        Hr2 += 0.5*einsum('mnij,mnab->ijab', imds.Woooo, r2)
        Hr2 += 0.5*einsum('abef,ijef->ijab', imds.Wvvvv, r2)

        vector = self.amplitudes_to_vector_ee(Hr1, Hr2)
        return vector

    ipccsd = rccsd_slow.RCCSD.ipccsd
    eaccsd = rccsd_slow.RCCSD.eaccsd
    eeccsd = rccsd_slow.RCCSD.eeccsd

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

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, **kwargs):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                        ccsd_lambda.kernel(self, eris, t1, t2, l1, l2, self.max_cycle, self.conv_tol_normt, self.verbose, make_intermediates, update_lambda)
        return self.l1, self.l2

    def ccsd_t_slow(self, t1=None, t2=None, eris=None):
        if eris is None: eris = self.ao2mo()
        if t1 is None or t2 is None:
            t1, t2 = self.t1, self.t2

        bcei = eris.ovvv.conj().transpose(3,2,1,0)
        majk = eris.ooov.conj().transpose(2,3,0,1)
        bcjk = eris.oovv.conj().transpose(2,3,0,1)

        fvo = eris.fov.transpose(1,0).conj()

        eia = eris.eia
        d3 = eris.get_eijkabc()

        t3c =(einsum('jkae,bcei->ijkabc', t2, bcei)
            - einsum('imbc,majk->ijkabc', t2, majk))
        t3c = t3c - t3c.transpose(0,1,2,4,3,5) - t3c.transpose(0,1,2,5,4,3)
        t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
        t3c /= d3

        t3d = einsum('ia,bcjk->ijkabc', t1, bcjk)
        t3d += einsum('ai,jkbc->ijkabc', fvo, t2)
        t3d = t3d - t3d.transpose(0,1,2,4,3,5) - t3d.transpose(0,1,2,5,4,3)
        t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
        t3d /= d3
        et = einsum('ijkabc,ijkabc,ijkabc', (t3c+t3d).conj(), d3, t3c) / 36
        return et

    def ccsd_t(self, t1=None, t2=None, eris=None, slice_size=None, free_vvvv=False):
        if eris is None: eris = self.ao2mo()
        if free_vvvv: eris.vvvv = None
        if t1 is None or t2 is None:
            t1, t2 = self.t1, self.t2
        nocc, nvir = t1.shape

        if slice_size is None:
            slice_size = nvir**4 / 3
        else:
            slice_size = slice_size / 3 * 1.25e5

        vir_blksize = min(nvir, max(4, int(((slice_size)**(1.0/3)/nocc))))
        logger.info(self, "nvir=%i, virtual blksize=%i", nvir, vir_blksize)
        if vir_blksize == nvir:
            return self.ccsd_t_slow(t1, t2, eris)
        bcei = eris.ovvv.conj().transpose(3,2,1,0)
        majk = eris.ooov.conj().transpose(2,3,0,1)
        bcjk = eris.oovv.conj().transpose(2,3,0,1)
        fvo = eris.fov.transpose(1,0).conj()

        mo_e = eris.mo_energy
        mo_e_o = asarray(mo_e[:nocc])
        mo_e_v = asarray(mo_e[nocc:])
        eijk = mo_e_o.reshape(nocc,1,1) + mo_e_o.reshape(1,nocc,1) + mo_e_o.reshape(1,1,nocc)
        eabc = mo_e_v.reshape(nvir,1,1) + mo_e_v.reshape(1,nvir,1) + mo_e_v.reshape(1,1,nvir)

        tasks = []
        for a0, a1 in lib.prange(0, nvir, vir_blksize):
            for b0, b1 in lib.prange(0, nvir, vir_blksize):
                for c0, c1 in lib.prange(0, nvir, vir_blksize):
                    if b0>=a0 and c0>=b0:
                        tasks.append((a0,a1,b0,b1,c0,c1))

        def get_w(a0, a1, b0, b1, c0, c1):
            w  = einsum('jkae,bcei->ijkabc', t2[:,:,a0:a1,:], bcei[b0:b1,c0:c1])
            w -= einsum('imbc,majk->ijkabc', t2[:,:,b0:b1,c0:c1], majk[:,a0:a1])
            return w

        def get_v(a0, a1, b0, b1, c0, c1):
            v  = einsum('ia,bcjk->ijkabc', t1[:,a0:a1], bcjk[b0:b1,c0:c1])
            v += einsum('ai,jkbc->ijkabc', fvo[a0:a1], t2[:,:,b0:b1,c0:c1])
            return v

        def get_pwv(a0,a1,b0,b1,c0,c1):
            d3 = eijk.reshape(nocc,nocc,nocc,1,1,1) - \
                 eabc[a0:a1,b0:b1,c0:c1].reshape(1,1,1,a1-a0, b1-b0, c1-c0)
            t3c = get_w(a0,a1,b0,b1,c0,c1) -\
                  get_w(b0,b1,a0,a1,c0,c1).transpose(0,1,2,4,3,5) -\
                  get_w(c0,c1,b0,b1,a0,a1).transpose(0,1,2,5,4,3)
            t3c = t3c - t3c.transpose(1,0,2,3,4,5) - t3c.transpose(2,1,0,3,4,5)
            t3c /= d3

            t3d = get_v(a0,a1,b0,b1,c0,c1) -\
                  get_v(b0,b1,a0,a1,c0,c1).transpose(0,1,2,4,3,5) -\
                  get_v(c0,c1,b0,b1,a0,a1).transpose(0,1,2,5,4,3)
            t3d = t3d - t3d.transpose(1,0,2,3,4,5) - t3d.transpose(2,1,0,3,4,5)
            t3d /= d3
            return t3c, t3d, d3

        et = 0
        for (a0, a1, b0, b1, c0, c1) in tasks:
            t3c, t3d, d3 = get_pwv(a0, a1, b0, b1, c0, c1)
            if a0 == c0:
                fac = 1
            elif a0 == b0 or b0 == c0:
                fac = 3
            else:
                fac = 6
            et += einsum('ijkabc,ijkabc', (t3c+t3d).conj(), d3*t3c) / 36 * fac
        return et

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        raise NotImplementedError

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        raise NotImplementedError

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds

        if isinstance(ipccsd_evecs, (tuple, list)):
            assert (len(ipccsd_evecs) == len(lipccsd_evecs))
            assert (ipccsd_evecs[0].size == lipccsd_evecs[0].size)
            nroots = len(ipccsd_evecs)
        else:
            assert (ipccsd_evecs.shape==lipccsd_evecs.shape)
            nroots = ipccsd_evecs.size // self.nip()
            ipccsd_evecs = ipccsd_evecs.reshape(-1, self.nip())
            lipccsd_evecs = lipccsd_evecs.reshape(-1, self.nip())

        t1, t2 = imds.t1, imds.t2
        eris = imds.eris
        assert (isinstance(eris, _PhysicistsERIs))

        oovv = eris.oovv
        ovvv = eris.ovvv
        ovov = eris.ovov

        ooov = eris.ooov
        vooo = ooov.conj().transpose(3,2,1,0)
        vvvo = ovvv.conj().transpose(3,2,1,0)
        oooo = eris.oooo

        # Create denominator
        eijkab = eris.get_eijkab()

        # Permutation operators
        def pijk(tmp):
            '''P(ijk)'''
            return tmp + tmp.transpose(1,2,0,3,4) + tmp.transpose(2,0,1,3,4)

        def pab(tmp):
            '''P(ab)'''
            return tmp - tmp.transpose(0,1,2,4,3)

        def pij(tmp):
            '''P(ij)'''
            return tmp - tmp.transpose(1,0,2,3,4)

        e_star = []
        ipccsd_evals = np.asarray(ipccsd_evals)
        for i in range(nroots):
            l1, l2 = self.vector_to_amplitudes_ip(lipccsd_evecs[i], **kwargs)
            r1, r2 = self.vector_to_amplitudes_ip(ipccsd_evecs[i], **kwargs)
            ldotr = dot(l1.ravel(), r1.ravel()) + .5* dot(l2.ravel(), r2.ravel())

            logger.info(self, 'Left-right amplitude overlap : %14.8e', ldotr)
            if abs(ldotr) < 1e-7:
                logger.warn(self, 'Small %s left-right amplitude overlap. Results '
                                 'may be inaccurate.', ldotr)

            l1 /= ldotr
            l2 /= ldotr

            # Denominator + eigenvalue(IP-CCSD)
            ip_eval = ipccsd_evals[i]
            denom = eijkab + ip_eval
            denom = 1. / denom

            tmp = einsum('ijab,k->ijkab', oovv, l1)
            lijkab = pijk(tmp)
            tmp = -einsum('jima,mkb->ijkab', ooov, l2)
            tmp = pijk(tmp)
            lijkab += pab(tmp)
            tmp = einsum('ieab,jke->ijkab', ovvv, l2)
            lijkab += pijk(tmp)

            tmp = einsum('mbke,m->bke', ovov, r1)
            tmp = einsum('bke,ijae->ijkab', tmp, t2)
            tmp = pijk(tmp)
            rijkab = -pab(tmp)
            tmp = einsum('mnjk,n->mjk', oooo, r1)
            tmp = einsum('mjk,imab->ijkab', tmp, t2)
            rijkab += pijk(tmp)
            tmp = einsum('amij,mkb->ijkab', vooo, r2)
            tmp = pijk(tmp)
            rijkab -= pab(tmp)
            tmp = einsum('baei,jke->ijkab', vvvo, r2)
            rijkab += pijk(tmp)

            deltaE = (1. / 12) * einsum('ijkab,ijkab->', lijkab, rijkab*denom)
            deltaE = deltaE.real
            logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        ip_eval + deltaE, deltaE)
            e_star.append(ip_eval + deltaE)
        return e_star

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, **kwargs):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds

        if isinstance(eaccsd_evecs, (tuple, list)):
            assert (len(eaccsd_evecs) == len(leaccsd_evecs))
            assert (eaccsd_evecs[0].size == leaccsd_evecs[0].size)
            nroots = len(eaccsd_evecs)
        else:
            assert (eaccsd_evecs.shape==leaccsd_evecs.shape)
            nroots = eaccsd_evecs.size // self.nea()
            eaccsd_evecs = eaccsd_evecs.reshape(-1, self.nea())
            leaccsd_evecs = leaccsd_evecs.reshape(-1, self.nea())

        t1, t2 = imds.t1, imds.t2
        eris = imds.eris
        assert (isinstance(eris, _PhysicistsERIs))

        vvvv = eris.vvvv
        oovv = eris.oovv
        ovvv = eris.ovvv
        ovov = eris.ovov
        ooov = eris.ooov
        vooo = ooov.conj().transpose(3,2,1,0)
        vvvo = ovvv.conj().transpose(3,2,1,0)

        # Create denominator

        eijabc = eris.get_eijabc()

        # Permutation operators
        def pabc(tmp):
            '''P(abc)'''
            return tmp + tmp.transpose(0,1,3,4,2) + tmp.transpose(0,1,4,2,3)

        def pij(tmp):
            '''P(ij)'''
            return tmp - tmp.transpose(1,0,2,3,4)

        def pab(tmp):
            '''P(ab)'''
            return tmp - tmp.transpose(0,1,3,2,4)


        e_star = []
        eaccsd_evals = np.asarray(eaccsd_evals)
        for i in range(nroots):
        #for ea_eval, ea_evec, ea_levec in zip(eaccsd_evals, eaccsd_evecs, leaccsd_evecs):
            # Enforcing <L|R> = 1
            l1, l2 = self.vector_to_amplitudes_ea(leaccsd_evecs[i], **kwargs)
            r1, r2 = self.vector_to_amplitudes_ea( eaccsd_evecs[i], **kwargs)
            ldotr = dot(l1.ravel(), r1.ravel()) + 0.5 * dot(l2.ravel(), r2.ravel())

            logger.info(self, 'Left-right amplitude overlap : %14.8e', ldotr)
            if abs(ldotr) < 1e-7:
                logger.warn(eom, 'Small %s left-right amplitude overlap. Results '
                                 'may be inaccurate.', ldotr)

            l1 /= ldotr
            l2 /= ldotr
            ea_eval = eaccsd_evals[i]
            # Denominator + eigenvalue(EA-CCSD)
            denom = eijabc + ea_eval
            denom = 1. / denom

            tmp = einsum('c,ijab->ijabc', l1, oovv)
            lijabc = -pabc(tmp)
            tmp = einsum('jima,mbc->ijabc', ooov, l2)
            lijabc += -pabc(tmp)
            tmp = einsum('ieab,jce->ijabc', ovvv, l2)
            tmp = pabc(tmp)
            lijabc += -pij(tmp)

            tmp = einsum('bcef,f->bce', vvvv, r1)
            tmp = einsum('bce,ijae->ijabc', tmp, t2)
            rijabc = -pabc(tmp)
            tmp = einsum('mcje,e->mcj', ovov, r1)
            tmp = einsum('mcj,imab->ijabc', tmp, t2)
            tmp = pabc(tmp)
            rijabc += pij(tmp)
            tmp = einsum('amij,mcb->ijabc', vooo, r2)
            rijabc += pabc(tmp)
            tmp = einsum('baei,jce->ijabc', vvvo, r2)
            tmp = pabc(tmp)
            rijabc -= pij(tmp)

            deltaE = (1. / 12) * einsum('ijabc,ijabc', lijabc, rijabc*denom)
            deltaE = deltaE.real
            logger.info(self, "Exc. energy, delta energy = %16.12f, %16.12f",
                        ea_eval + deltaE, deltaE)
            e_star.append(ea_eval + deltaE)

        return e_star

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocc = eris.nocc
    eris.foo = eris.fock[:nocc,:nocc]
    eris.fov = eris.fock[:nocc,nocc:]
    eris.fvv = eris.fock[nocc:,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eris._foo  = np.diag(np.diag(eris.foo))
    eris._fvv  = np.diag(np.diag(eris.fvv))
    eris.eia = mo_e_o[:,None] - mo_e_v
    eris.eijab = lib.direct_sum('ia,jb->ijab', eris.eia, eris.eia)

    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert(eris.mo_coeff.dtype == np.double)
        mo_a = eris.mo_coeff[:nao//2]
        mo_b = eris.mo_coeff[nao//2:]
        orbspin = eris.orbspin
        if orbspin is None:
            eri  = ao2mo.kernel(mycc._scf._eri, mo_a)
            eri += ao2mo.kernel(mycc._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mycc._scf._eri, (mo_a,mo_a,mo_b,mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mycc._scf._eri, mo)
            if eri.size == nmo**4:  # if mycc._scf._eri is a complex array
                sym_forbid = (orbspin[:,None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:,None] != orbspin)[np.tril_indices(nmo)]
            eri[sym_forbid,:] = 0
            eri[:,sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

class _PhysicistsERIs(gccsd._PhysicistsERIs):

    def get_eijkabc(self):
        nocc, nvir = self.eia.shape
        return self.eijab.reshape(nocc,nocc,1,nvir,nvir,1) + self.eia.reshape(1,1,nocc,1,1,nvir)

    def get_eijkab(self):
        nocc, nvir = self.eia.shape
        mo_e_o = asarray(self.mo_energy[:self.nocc])
        eijkab = self.eijab.reshape(nocc,nocc,1,nvir,nvir) + mo_e_o.reshape(1,1,nocc,1,1)
        return eijkab

    def get_eijabc(self):
        nocc, nvir = self.eia.shape
        mo_e_v = asarray(self.mo_energy[self.nocc:])
        eijabc = self.eijab.reshape(nocc,nocc,nvir,nvir,1) - mo_e_v.reshape(1,1,1,1,nvir)
        return eijabc


class _IMDS(eom_gccsd._IMDS):

    def __init__(self, cc, eris=None):
        eom_gccsd._IMDS.__init__(self, cc, eris)
        self.made_t3p2_ip_imds = False
        self.made_t3p2_ea_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo = imd.Foo(t1, t2, eris)
        self.Fvv = imd.Fvv(t1, t2, eris)
        self.Fov = imd.Fov(t1, t2, eris)

        # 2 virtuals
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-CCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo = imd.Woooo(t1, t2, eris)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD IP intermediates', *cput0)
        return self

    def make_t3p2_ip(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared = False  # Force update
        self.make_ip()  # Make after t1/t2 updated
        self.Wovoo = self.Wovoo + Wovoo

        self.made_ip_imds = self.made_t3p2_ip_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (time.clock(), time.time())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        self.Wvvvv = imd.Wvvvv(t1, t2, eris)
        self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD EA intermediates', *cput0)
        return self

    def make_t3p2_ea(self, cc):
        cput0 = (time.clock(), time.time())

        t1, t2, eris = cc.t1, cc.t2, self.eris
        delta_E_corr, pt1, pt2, Wovoo, Wvvvo = \
            imd.get_t3p2_imds_slow(cc, t1, t2, eris)
        self.t1 = pt1
        self.t2 = pt2

        self._made_shared = False  # Force update
        self.make_ea()  # Make after t1/t2 updated
        self.Wvvvo = self.Wvvvo + Wvvvo

        self.made_ea_imds = self.made_t3p2_ea_imds = True
        logger.timer_debug1(self, 'EOM-CCSD(T)a EA intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo = imd.Woooo(t1, t2, eris)
            self.Wooov = imd.Wooov(t1, t2, eris)
            self.Wovoo = imd.Wovoo(t1, t2, eris)

        elif self.made_t3p2_ip_imds:
            logger.warn(self, 'imds.t1/t2/Wovoo is detected to carry t3p2 contribution. '
                        'This could be due to previous execution of ipccsd_t_star.\n'
                        'Please make sure this is the expected behavior, otherwise please do self.imds=None')

        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(t1, t2, eris)
            self.Wvvvv = imd.Wvvvv(t1, t2, eris)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris, self.Wvvvv)

        elif self.made_t3p2_ea_imds:
            logger.warn(self, 'imds.t1/t2/Wvvvo is detected to carry t3p2 contribution. '
                        'This could be due to previous execution of eaccsd_t_star.\n'
                        'Please make sure this is the expected behavior, otherwise please do self.imds=None')

        self.made_ee_imds = True
        logger.timer(self, 'EOM-CCSD EE intermediates', *cput0)
        return self

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    # Freeze 1s electrons
    frozen = [0,1,2,3]
    gcc = GCCSD(mf, frozen=frozen)
    ecc, t1, t2 = gcc.kernel()
    print(ecc - -0.3486987472235819)

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build()
    mf = scf.UHF(mol).run()
    mf = scf.addons.convert_to_ghf(mf)

    mycc = GCCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)

    l1, l2 = mycc.solve_lambda(t1=t1, t2=t2)
    # This test can fail due to potential rotation of MO
    print(lib.finger(l1) - 0.0008702886643046741)
    print(lib.finger(l2) - -0.5497018043557017)

    et = mycc.ccsd_t_slow()
    print(et--0.003060022117815608)

    et = mycc.ccsd_t()
    print(et--0.003060022117815608)

    e,vipl = mycc.ipccsd(nroots=8, left=True)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    e,vipr = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    estar = mycc.ipccsd_star_contract(e, vipr, vipl)

    print(estar[0] - 0.43793206746897545)
    print(estar[2] - 0.5228702567036676)
    print(estar[4] - 0.6799456669820577)

    e,vear = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    e,veal = mycc.eaccsd(nroots=8, left=True)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)

    estar = mycc.eaccsd_star_contract(e, vear, veal)
    print(estar[0] - 0.16656253472780994)
    print(estar[2] - 0.23944154865211192)
    print(estar[4] - 0.41399418895492107)

    e,v = mycc.eeccsd(nroots=4)
    print(e[0] - 0.2757159395886167)
    print(e[1] - 0.2757159395886167)
    print(e[2] - 0.2757159395886167)
    print(e[3] - 0.3005716731825082)

    e,v = mycc.ipccsd_t_star(nroots=8)
    print(e[0] - 0.43455703)
    print(e[2] - 0.51991377)
    print(e[4] - 0.67944506)

    e, v =mycc.eaccsd_t_star(nroots=8)
    print(e[0] - 0.16785705)
    print(e[2] - 0.2409836)
    print(e[4] - 0.51127648)
