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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#         Yang Gao <younggao1994@gmail.com>

'''
UCCSD with spatial integrals
'''

import time
from functools import reduce
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd, uccsd, eom_uccsd
from pyscf import scf
from pyscf.cc import uintermediates_slow as imd
from pyscf.lib import linalg_helper

einsum = lib.einsum
# This is unrestricted (U)CCSD, in spatial-orbital form.

def update_amps(mycc, t1, t2, eris):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    Fvv, FVV = imd.cc_Fvv(t1, t2, eris)
    Foo, FOO = imd.cc_Foo(t1, t2, eris)
    Fov, FOV = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Fvv -= eris._fvv
    FVV -= eris._fVV
    Foo -= eris._foo
    FOO -= eris._fOO

    # T1 equation
    Ht1a = eris.fov.conj().copy()
    Ht1b = eris.fOV.conj().copy()

    Ht1a += einsum('imae,me->ia', t2aa, Fov)
    Ht1a += einsum('imae,me->ia', t2ab, FOV)
    Ht1b += einsum('imae,me->ia', t2bb, FOV)
    Ht1b += einsum('miea,me->ia', t2ab, Fov)

    Ht1a += einsum('ie,ae->ia', t1a, Fvv)
    Ht1b += einsum('ie,ae->ia', t1b, FVV)
    Ht1a -= einsum('ma,mi->ia', t1a, Foo)
    Ht1b -= einsum('ma,mi->ia', t1b, FOO)

    ovoo = eris.ooov.transpose(2,3,0,1) - eris.ooov.transpose(0,3,2,1)
    Ht1a += 0.5*einsum('mnae,meni->ia', t2aa, ovoo)
    OVOO = eris.OOOV.transpose(2,3,0,1) - eris.OOOV.transpose(0,3,2,1)
    Ht1b += 0.5*einsum('mnae,meni->ia', t2bb, OVOO)


    Ht1a -= einsum('nmae,nime->ia', t2ab, eris.ooOV)
    Ht1b -= einsum('mnea,nime->ia', t2ab, eris.OOov)

    Ht1a += einsum('mf,aimf->ia', t1a, eris.voov)
    Ht1a -= einsum('mf,miaf->ia', t1a, eris.oovv)
    Ht1a += einsum('mf,aimf->ia', t1b, eris.voOV)

    Ht1b += einsum('mf,aimf->ia', t1b, eris.VOOV)
    Ht1b -= einsum('mf,miaf->ia', t1b, eris.OOVV)
    Ht1b += einsum('mf,fmia->ia', t1a, eris.voOV.conj())

    Ht1a += einsum('imef,fmea->ia', t2aa, eris.vovv.conj())
    Ht1a += einsum('imef,fmea->ia', t2ab, eris.VOvv.conj())
    Ht1b += einsum('imef,fmea->ia', t2bb, eris.VOVV.conj())
    Ht1b += einsum('mife,fmea->ia', t2ab, eris.voVV.conj())

    Ftmpa = Fvv - 0.5 * einsum('mb,me->be', t1a, Fov)
    Ftmpb = FVV - 0.5 * einsum('mb,me->be', t1b, FOV)

    # T2 equation
    Ht2aa = einsum('ijae,be->ijab', t2aa, Ftmpa)
    Ht2bb = einsum('ijae,be->ijab', t2bb, Ftmpb)
    Ht2ab = einsum('ijae,be->ijab', t2ab, Ftmpb)
    Ht2ab += einsum('ijeb,ae->ijab', t2ab, Ftmpa)

    #P(ab)
    Ht2aa -= einsum('ijbe,ae->ijab', t2aa, Ftmpa)
    Ht2bb -= einsum('ijbe,ae->ijab', t2bb, Ftmpb)

    # Foo equation
    Ftmpa = Foo + 0.5 * einsum('je,me->mj', t1a, Fov)
    Ftmpb = FOO + 0.5 * einsum('je,me->mj', t1b, FOV)

    Ht2aa -= einsum('imab,mj->ijab', t2aa, Ftmpa)
    Ht2bb -= einsum('imab,mj->ijab', t2bb, Ftmpb)
    Ht2ab -= einsum('imab,mj->ijab', t2ab, Ftmpb)
    Ht2ab -= einsum('mjab,mi->ijab', t2ab, Ftmpa)

    #P(ij)
    Ht2aa += einsum('jmab,mi->ijab', t2aa, Ftmpa)
    Ht2bb += einsum('jmab,mi->ijab', t2bb, Ftmpb)

    Ht2aa += (eris.ovov.transpose(0,2,1,3) - eris.ovov.transpose(2,0,1,3)).conj()
    Ht2bb += (eris.OVOV.transpose(0,2,1,3) - eris.OVOV.transpose(2,0,1,3)).conj()
    Ht2ab += eris.ovOV.transpose(0,2,1,3).conj()

    tauaa, tauab, taubb = imd.make_tau(t2, t1, t1)
    Woooo, WooOO, WOOOO = imd.cc_Woooo(t1, t2, eris)

    Woooo += .5 * einsum('menf,ijef->minj', eris.ovov, tauaa)
    WOOOO += .5 * einsum('menf,ijef->minj', eris.OVOV, taubb)
    WooOO += .5 * einsum('menf,ijef->minj', eris.ovOV, tauab)

    Ht2aa += einsum('minj,mnab->ijab', Woooo, tauaa) * .5
    Ht2bb += einsum('minj,mnab->ijab', WOOOO, taubb) * .5
    Ht2ab += einsum('minj,mnab->ijab', WooOO, tauab)

    # add_vvvv block
    imd.cc_add_vvvv(t1, t2, eris, Ht2aa, Ht2ab, Ht2bb)

    Wovvo, WovVO, WOVvo, WOVVO, WoVVo, WOvvO = \
        imd.cc_Wovvo(t1, t2, eris)

    Ht2ab += einsum('imae,mebj->ijab', t2aa, WovVO)
    Ht2ab += einsum('imae,mebj->ijab', t2ab, WOVVO)
    Ht2ab -= einsum('ie,ma,emjb->ijab', t1a, t1a, eris.voOV.conj())

    Ht2ab += einsum('miea,mebj->jiba', t2ab, Wovvo)
    Ht2ab += einsum('miea,mebj->jiba', t2bb, WOVvo)

    Ht2ab -= einsum('ie,ma,bjme->jiba', t1b, t1b, eris.voOV)
    Ht2ab += einsum('imea,mebj->ijba', t2ab, WOvvO)
    Ht2ab -= einsum('ie,ma,mjbe->ijba', t1a, t1b, eris.OOvv)
    Ht2ab += einsum('miae,mebj->jiab', t2ab, WoVVo)
    Ht2ab -= einsum('ie,ma,mjbe->jiab', t1b, t1a, eris.ooVV)


    u2aa = einsum('imae,mebj->ijab', t2aa, Wovvo)
    u2aa += einsum('imae,mebj->ijab', t2ab, WOVvo)
    u2aa += einsum('ie,ma,mjbe->ijab',t1a,t1a,eris.oovv)
    u2aa -= einsum('ie,ma,bjme->ijab',t1a,t1a,eris.voov)

    u2aa += einsum('ie,bjae->ijab', t1a, eris.vovv)
    u2aa -= einsum('ma,imjb->ijab', t1a, eris.ooov.conj())

    u2aa = u2aa - u2aa.transpose(1,0,2,3)
    u2aa = u2aa - u2aa.transpose(0,1,3,2)
    Ht2aa += u2aa
    del u2aa, WOvvO, WoVVo, Wovvo, WOVvo

    u2bb = einsum('imae,mebj->ijab', t2bb, WOVVO)
    u2bb += einsum('miea,mebj->ijab', t2ab,WovVO)
    u2bb += einsum('ie,ma,mjbe->ijab',t1b, t1b, eris.OOVV)
    u2bb -= einsum('ie,ma,bjme->ijab',t1b, t1b, eris.VOOV)
    u2bb += einsum('ie,bjae->ijab', t1b, eris.VOVV)
    u2bb -= einsum('ma,imjb->ijab', t1b, eris.OOOV.conj())

    u2bb = u2bb - u2bb.transpose(1,0,2,3)
    u2bb = u2bb - u2bb.transpose(0,1,3,2)
    Ht2bb += u2bb
    del u2bb, WOVVO, WovVO

    Ht2ab += einsum('ie,bjae->ijab', t1a, eris.VOvv)
    Ht2ab += einsum('je,aibe->ijab', t1b, eris.voVV)
    Ht2ab -= einsum('ma,imjb->ijab', t1a, eris.ooOV.conj())
    Ht2ab -= einsum('mb,jmia->ijab', t1b, eris.OOov.conj())

    Ht1a /= eris.eia
    Ht1b /= eris.eIA

    Ht2aa /= eris.eijab
    Ht2ab /= eris.eiJaB
    Ht2bb /= eris.eIJAB

    time0 = log.timer_debug1('update t1 t2', *time0)
    return (Ht1a, Ht1b), (Ht2aa, Ht2ab, Ht2bb)

def energy(mycc, t1=None, t2=None, eris=None, fac=1.0):
    '''UCCSD correlation energy'''
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    e = einsum('ia,ia->', eris.fov, t1a)
    e+= einsum('ia,ia->', eris.fOV, t1b)

    tauaa = t2aa + 2*einsum('ia,jb->ijab', t1a, t1a)
    tauab = t2ab +   einsum('ia,jb->ijab', t1a, t1b)
    taubb = t2bb + 2*einsum('ia,jb->ijab', t1b, t1b)

    e += 0.25*(einsum('iajb,ijab->',eris.ovov,tauaa)
             - einsum('jaib,ijab->',eris.ovov,tauaa))

    e += einsum('iajb,ijab->',eris.ovOV,tauab)
    e += 0.25*(einsum('iajb,ijab->',eris.OVOV,taubb)
             - einsum('jaib,ijab->',eris.OVOV,taubb))
    e *= fac
    if abs(e.imag) > 1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in UCCSD energy %s', e)
    return e.real

def init_amps(mycc, eris=None):
    time0 = time.clock(), time.time()
    if eris is None:
        eris = mycc.ao2mo(self.mo_coeff)

    t1a = eris.fov.conj() / eris.eia
    t1b = eris.fOV.conj() / eris.eIA
    t2aa = eris.ovov.conj().transpose(0,2,1,3) / eris.eijab
    t2aa-= t2aa.transpose(0,1,3,2)
    t2ab = eris.ovOV.conj().transpose(0,2,1,3) / eris.eiJaB
    t2bb = eris.OVOV.conj().transpose(0,2,1,3) / eris.eIJAB
    t2bb-= t2bb.transpose(0,1,3,2)

    e  =      einsum('ijab,iajb->', t2ab, eris.ovOV)
    e += 0.25*einsum('ijab,iajb->', t2aa, eris.ovov)
    e -= 0.25*einsum('ijab,ibja->', t2aa, eris.ovov)
    e += 0.25*einsum('ijab,iajb->', t2bb, eris.OVOV)
    e -= 0.25*einsum('ijab,ibja->', t2bb, eris.OVOV)

    t1 = (t1a, t1b)
    t2 = (t2aa, t2ab, t2bb)

    logger.timer(mycc, 'init mp2', *time0)
    return e, t1, t2

def uccsd_t_slow(mycc, eris, t1=None, t2=None):
    if t1 is None or t2 is None:
        t1, t2 = mycc.t1, mycc.t2

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) + \
                t.transpose(2,0,1,5,3,4) + t.transpose(0,2,1,3,5,4) + \
                t.transpose(2,1,0,5,4,3) + t.transpose(1,0,2,4,3,5))
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

    # aaa
    d3 = eris.get_eijkabc('aaa')

    w = einsum('ijae,ckbe->ijkabc', t2aa, eris.vovv)
    w-= einsum('mkbc,jmia->ijkabc', t2aa, eris.ooov.conj())
    r = r6(w)
    v = einsum('jbkc,ia->ijkabc', eris.ovov.conj(), t1a)
    v+= einsum('jkbc,ai->ijkabc', t2aa, fvo) * .5
    wvd = p6(w + v) / d3
    et = einsum('ijkabc,ijkabc', wvd.conj(), r)

    # bbb
    d3 = eris.get_eijkabc('bbb')

    w = einsum('ijae,ckbe->ijkabc', t2bb, eris.VOVV)
    w-= einsum('imab,jmkc->ijkabc', t2bb, eris.OOOV.conj())
    r = r6(w)
    v = einsum('jbkc,ia->ijkabc', eris.OVOV.conj(), t1b)
    v+= einsum('jkbc,ai->ijkabc', t2bb, fVO) * .5
    wvd = p6(w + v) / d3
    et += einsum('ijkabc,ijkabc', wvd.conj(), r)

    # baa
    w  = einsum('jiea,ckbe->ijkabc', t2ab, eris.vovv) * 2
    w += einsum('jibe,ckae->ijkabc', t2ab, eris.voVV) * 2
    w += einsum('jkbe,aice->ijkabc', t2aa, eris.VOvv)
    w -= einsum('miba,jmkc->ijkabc', t2ab, eris.ooov.conj()) * 2
    w -= einsum('jmba,imkc->ijkabc', t2ab, eris.OOov.conj()) * 2
    w -= einsum('jmbc,kmia->ijkabc', t2aa, eris.ooOV.conj())
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = einsum('jbkc,ia->ijkabc', eris.ovov.conj(), t1b)
    v += einsum('kcia,jb->ijkabc', eris.ovOV.conj(), t1a)
    v += einsum('kcia,jb->ijkabc', eris.ovOV.conj(), t1a)
    v += einsum('jkbc,ai->ijkabc', t2aa, fVO) * .5
    v += einsum('kica,bj->ijkabc', t2ab, fvo) * 2
    w += v
    d3 = eris.get_eijkabc('baa')
    r /= d3
    et += einsum('ijkabc,ijkabc', w.conj(), r)

    # abb
    w  = einsum('ijae,ckbe->ijkabc', t2ab, eris.VOVV) * 2
    w += einsum('ijeb,ckae->ijkabc', t2ab, eris.VOvv) * 2
    w += einsum('jkbe,aice->ijkabc', t2bb, eris.voVV)
    w -= einsum('imab,jmkc->ijkabc', t2ab, eris.OOOV.conj()) * 2
    w -= einsum('mjab,imkc->ijkabc', t2ab, eris.ooOV.conj()) * 2
    w -= einsum('jmbc,kmia->ijkabc', t2bb, eris.OOov.conj())
    r = w - w.transpose(0,2,1,3,4,5)
    r = r + r.transpose(0,2,1,3,5,4)
    v  = einsum('jbkc,ia->ijkabc', eris.OVOV.conj(), t1a)
    v += einsum('iakc,jb->ijkabc', eris.ovOV.conj(), t1b)
    v += einsum('iakc,jb->ijkabc', eris.ovOV.conj(), t1b)
    v += einsum('jkbc,ai->ijkabc', t2bb, fvo) * .5
    v += einsum('ikac,bj->ijkabc', t2ab, fVO) * 2
    w += v

    d3 = eris.get_eijkabc('abb')
    r /= d3
    et += einsum('ijkabc,ijkabc', w.conj(), r)

    et *= .25
    return et

def make_tau(t2, t1, r1, fac=1):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1aa = make_tau_aa(t2[0], t1a, r1a, fac)
    tau1bb = make_tau_aa(t2[2], t1b, r1b, fac)
    tau1ab = make_tau_ab(t2[1], t1, r1, fac)
    return tau1aa, tau1ab, tau1bb

def make_tau_aa(t2aa, t1a, r1a, fac=1):
    tau1aa = einsum('ia,jb->ijab', t1a, r1a)
    tau1aa-= einsum('ia,jb->jiab', t1a, r1a)
    tau1aa = tau1aa - tau1aa.transpose(0,1,3,2)
    tau1aa *= fac * .5
    tau1aa += t2aa
    return tau1aa

def make_tau_ab(t2ab, t1, r1, fac=1):
    t1a, t1b = t1
    r1a, r1b = r1
    tau1ab = t2ab + einsum('ia,jb->ijab', t1a, r1b) *fac*.5 +\
             einsum('ia,jb->ijab', r1a, t1b) * fac*.5
    return tau1ab

def make_intermediates(mycc, t1, t2, eris):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2

    fooa = eris.foo
    fova = eris.fov
    fvoa = eris.fov.conj().transpose(1,0)
    fvva = eris.fvv
    foob = eris.fOO
    fovb = eris.fOV
    fvob = eris.fOV.conj().transpose(1,0)
    fvvb = eris.fVV

    tauaa, tauab, taubb = make_tau(t2, t1, t1)


    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    ovOV = eris.ovOV

    v1a  = fvva - einsum('ja,jb->ba', fova, t1a)
    v1b  = fvvb - einsum('ja,jb->ba', fovb, t1b)
    v1a += einsum('jcka,jkbc->ba', ovov, tauaa) * .5
    v1a -= einsum('jakc,jkbc->ba', ovOV, tauab) * .5
    v1a -= einsum('kajc,kjbc->ba', ovOV, tauab) * .5
    v1b += einsum('jcka,jkbc->ba', OVOV, taubb) * .5
    v1b -= einsum('kcja,kjcb->ba', ovOV, tauab) * .5
    v1b -= einsum('jcka,jkcb->ba', ovOV, tauab) * .5

    v2a  = fooa + einsum('ib,jb->ij', fova, t1a)
    v2b  = foob + einsum('ib,jb->ij', fovb, t1b)
    v2a += einsum('ibkc,jkbc->ij', ovov, tauaa) * .5
    v2a += einsum('ibkc,jkbc->ij', ovOV, tauab)
    v2b += einsum('ibkc,jkbc->ij', OVOV, taubb) * .5
    v2b += einsum('kcib,kjcb->ij', ovOV, tauab)

    ovoo = eris.ooov.transpose(2,3,0,1) - eris.ooov.transpose(0,3,2,1)
    OVOO = eris.OOOV.transpose(2,3,0,1) - eris.OOOV.transpose(0,3,2,1)
    OVoo = eris.ooOV.transpose(2,3,0,1)
    ovOO = eris.OOov.transpose(2,3,0,1)
    v2a -= einsum('ibkj,kb->ij', ovoo, t1a)
    v2a += einsum('kbij,kb->ij', OVoo, t1b)
    v2b -= einsum('ibkj,kb->ij', OVOO, t1b)
    v2b += einsum('kbij,kb->ij', ovOO, t1a)

    v5a  = fvoa + einsum('kc,jkbc->bj', fova, t2aa)
    v5a += einsum('kc,jkbc->bj', fovb, t2ab)
    v5b  = fvob + einsum('kc,jkbc->bj', fovb, t2bb)
    v5b += einsum('kc,kjcb->bj', fova, t2ab)
    tmp  = fova - einsum('kdlc,ld->kc', ovov, t1a)
    tmp += einsum('kcld,ld->kc', ovOV, t1b)
    v5a += einsum('kc,kb,jc->bj', tmp, t1a, t1a)
    tmp  = fovb - einsum('kdlc,ld->kc', OVOV, t1b)
    tmp += einsum('ldkc,ld->kc', ovOV, t1a)
    v5b += einsum('kc,kb,jc->bj', tmp, t1b, t1b)
    v5a -= einsum('lckj,klbc->bj', ovoo, t2aa) * .5
    v5a -= einsum('lckj,klbc->bj', OVoo, t2ab)
    v5b -= einsum('lckj,klbc->bj', OVOO, t2bb) * .5
    v5b -= einsum('lckj,lkcb->bj', ovOO, t2ab)

    oooo = eris.oooo
    OOOO = eris.OOOO
    ooOO = eris.ooOO
    woooo  = einsum('icjl,kc->ikjl', ovoo, t1a)
    wOOOO  = einsum('icjl,kc->ikjl', OVOO, t1b)
    wooOO  = einsum('icjl,kc->ikjl', ovOO, t1a)
    wooOO += einsum('jcil,kc->iljk', OVoo, t1b)
    woooo += (oooo - oooo.transpose(0,3,2,1)) * .5
    wOOOO += (OOOO - OOOO.transpose(0,3,2,1)) * .5
    wooOO += ooOO
    woooo += einsum('icjd,klcd->ikjl', ovov, tauaa) * .25
    wOOOO += einsum('icjd,klcd->ikjl', OVOV, taubb) * .25
    wooOO += einsum('icjd,klcd->ikjl', ovOV, tauab)

    v4ovvo  = einsum('jbld,klcd->jbck', ovov, t2aa)
    v4ovvo += einsum('jbld,klcd->jbck', ovOV, t2ab)
    v4ovvo += eris.voov.transpose(2,3,0,1)
    v4ovvo -= eris.oovv.transpose(0,3,2,1)
    v4OVVO  = einsum('jbld,klcd->jbck', OVOV, t2bb)
    v4OVVO += einsum('ldjb,lkdc->jbck', ovOV, t2ab)
    v4OVVO += eris.VOOV.transpose(2,3,0,1)
    v4OVVO -= eris.OOVV.transpose(0,3,2,1)
    v4OVvo  = einsum('ldjb,klcd->jbck', ovOV, t2aa)
    v4OVvo += einsum('jbld,klcd->jbck', OVOV, t2ab)
    v4OVvo += eris.voOV.transpose(2,3,0,1)
    v4ovVO  = einsum('jbld,klcd->jbck', ovOV, t2bb)
    v4ovVO += einsum('jbld,lkdc->jbck', ovov, t2ab)
    v4ovVO += eris.VOov.transpose(2,3,0,1)
    v4oVVo  = einsum('jdlb,kldc->jbck', ovOV, t2ab)
    v4oVVo -= eris.ooVV.transpose(0,3,2,1)
    v4OvvO  = einsum('lbjd,lkcd->jbck', ovOV, t2ab)
    v4OvvO -= eris.OOvv.transpose(0,3,2,1)

    woovo  = einsum('ibck,jb->ijck', v4ovvo, t1a)
    wOOVO  = einsum('ibck,jb->ijck', v4OVVO, t1b)
    wOOvo  = einsum('ibck,jb->ijck', v4OVvo, t1b)
    wOOvo -= einsum('ibck,jb->ikcj', v4OvvO, t1a)
    wooVO  = einsum('ibck,jb->ijck', v4ovVO, t1a)
    wooVO -= einsum('ibck,jb->ikcj', v4oVVo, t1b)
    woovo += ovoo.conj().transpose(3,2,1,0) * .5
    wOOVO += OVOO.conj().transpose(3,2,1,0) * .5
    wooVO += OVoo.conj().transpose(3,2,1,0)
    wOOvo += ovOO.conj().transpose(3,2,1,0)
    woovo -= einsum('iclk,jlbc->ikbj', ovoo, t2aa)
    woovo += einsum('lcik,jlbc->ikbj', OVoo, t2ab)
    wOOVO -= einsum('iclk,jlbc->ikbj', OVOO, t2bb)
    wOOVO += einsum('lcik,ljcb->ikbj', ovOO, t2ab)
    wooVO -= einsum('iclk,ljcb->ikbj', ovoo, t2ab)
    wooVO += einsum('lcik,jlbc->ikbj', OVoo, t2bb)
    wooVO -= einsum('iclk,jlcb->ijbk', ovOO, t2ab)
    wOOvo -= einsum('iclk,jlbc->ikbj', OVOO, t2ab)
    wOOvo += einsum('lcik,jlbc->ikbj', ovOO, t2aa)
    wOOvo -= einsum('iclk,ljbc->ijbk', OVoo, t2ab)

    wvvvo  = einsum('jack,jb->back', v4ovvo, t1a)
    wVVVO  = einsum('jack,jb->back', v4OVVO, t1b)
    wVVvo  = einsum('jack,jb->back', v4OVvo, t1b)
    wVVvo -= einsum('jack,jb->cabk', v4oVVo, t1a)
    wvvVO  = einsum('jack,jb->back', v4ovVO, t1a)
    wvvVO -= einsum('jack,jb->cabk', v4OvvO, t1b)
    wvvvo += einsum('lajk,jlbc->back', .25*ovoo, tauaa)
    wVVVO += einsum('lajk,jlbc->back', .25*OVOO, taubb)
    wVVvo -= einsum('lajk,jlcb->back', OVoo, tauab)
    wvvVO -= einsum('lajk,ljbc->back', ovOO, tauab)

    w3a  = einsum('jbck,jb->ck', v4ovvo, t1a)
    w3a += einsum('jbck,jb->ck', v4OVvo, t1b)
    w3b  = einsum('jbck,jb->ck', v4OVVO, t1b)
    w3b += einsum('jbck,jb->ck', v4ovVO, t1a)

    wovvo  = v4ovvo
    wOVVO  = v4OVVO
    wovVO  = v4ovVO
    wOVvo  = v4OVvo
    woVVo  = v4oVVo
    wOvvO  = v4OvvO
    wovvo += einsum('jbld,kd,lc->jbck', ovov, t1a, -t1a)
    wOVVO += einsum('jbld,kd,lc->jbck', OVOV, t1b, -t1b)
    wovVO += einsum('jbld,kd,lc->jbck', ovOV, t1b, -t1b)
    wOVvo += einsum('ldjb,kd,lc->jbck', ovOV, t1a, -t1a)
    woVVo += einsum('jdlb,kd,lc->jbck', ovOV, t1a,  t1b)
    wOvvO += einsum('lbjd,kd,lc->jbck', ovOV, t1b,  t1a)
    wovvo -= einsum('jblk,lc->jbck', ovoo, t1a)
    wOVVO -= einsum('jblk,lc->jbck', OVOO, t1b)
    wovVO -= einsum('jblk,lc->jbck', ovOO, t1b)
    wOVvo -= einsum('jblk,lc->jbck', OVoo, t1a)
    woVVo += einsum('lbjk,lc->jbck', OVoo, t1b)
    wOvvO += einsum('lbjk,lc->jbck', ovOO, t1a)

    ovvv = eris.vovv.transpose(1,0,3,2).conj() - eris.vovv.transpose(1,2,3,0).conj()
    v1a -= einsum('jabc,jc->ba', ovvv, t1a)
    v5a += einsum('kdbc,jkcd->bj', ovvv, t2aa) * .5
    woovo += einsum('idcb,kjbd->ijck', ovvv, tauaa) * .25
    wovvo += einsum('jbcd,kd->jbck', ovvv, t1a)
    wvvvo -= ovvv.conj().transpose(3,2,1,0) * .5
    wvvvo += einsum('jacd,kjbd->cabk', ovvv, t2aa)
    wvvVO += einsum('jacd,jkdb->cabk', ovvv, t2ab)
    ovvv = tmp = None

    OVVV = eris.VOVV.transpose(1,0,3,2).conj() - eris.VOVV.transpose(1,2,3,0).conj()
    v1b -= einsum('jabc,jc->ba', OVVV, t1b)
    v5b += einsum('kdbc,jkcd->bj', OVVV, t2bb) * .5
    wOOVO += einsum('idcb,kjbd->ijck', OVVV, taubb) * .25
    wOVVO += einsum('jbcd,kd->jbck', OVVV, t1b)
    wVVVO -= OVVV.conj().transpose(3,2,1,0) * .5
    wVVVO += einsum('jacd,kjbd->cabk', OVVV, t2bb)
    wVVvo += einsum('jacd,kjbd->cabk', OVVV, t2ab)
    OVVV = tmp = None

    OVvv = eris.VOvv.transpose(1,0,3,2)
    v1a += einsum('jcba,jc->ba', OVvv, t1b)
    v5a += einsum('kdbc,jkcd->bj', OVvv, t2ab)
    wOOvo += einsum('idcb,kjbd->ijck', OVvv, tauab)
    wOVvo += einsum('jbcd,kd->jbck', OVvv, t1a)
    wOvvO -= einsum('jdcb,kd->jbck', OVvv, t1b)
    wvvVO -= OVvv.conj().transpose(3,2,1,0)
    wvvvo -= einsum('kdca,jkbd->cabj', OVvv, t2ab)
    wvvVO -= einsum('kdca,jkbd->cabj', OVvv, t2bb)
    wVVvo += einsum('kacd,jkdb->bacj', OVvv, t2ab)
    OVvv = tmp = None

    ovVV = eris.voVV.transpose(1,0,3,2)
    v1b += einsum('jcba,jc->ba', ovVV, t1a)
    v5b += einsum('kdbc,kjdc->bj', ovVV, t2ab)
    wooVO += einsum('idcb,jkdb->ijck', ovVV, tauab)
    wovVO += einsum('jbcd,kd->jbck', ovVV, t1b)
    woVVo -= einsum('jdcb,kd->jbck', ovVV, t1a)
    wVVvo -= ovVV.conj().transpose(3,2,1,0)
    wVVVO -= einsum('kdca,kjdb->cabj', ovVV, t2ab)
    wVVvo -= einsum('kdca,jkbd->cabj', ovVV, t2aa)
    wvvVO += einsum('kacd,kjbd->bacj', ovVV, t2ab)
    ovVV = tmp = None

    w3a += v5a
    w3b += v5b
    w3a += einsum('cb,jb->cj', v1a, t1a)
    w3b += einsum('cb,jb->cj', v1b, t1b)
    w3a -= einsum('jk,jb->bk', v2a, t1a)
    w3b -= einsum('jk,jb->bk', v2b, t1b)

    class _LIMDS: pass
    imds = _LIMDS()
    imds.woooo = woooo
    imds.wOOOO = wOOOO
    imds.wooOO = wooOO
    imds.wovvo = wovvo
    imds.wOVVO = wOVVO
    imds.wovVO = wovVO
    imds.wOVvo = wOVvo
    imds.woVVo = woVVo
    imds.wOvvO = wOvvO
    imds.woovo = woovo
    imds.wOOVO = wOOVO
    imds.wOOvo = wOOvo
    imds.wooVO = wooVO
    imds.wvvvo = wvvvo
    imds.wVVVO = wVVVO
    imds.wVVvo = wVVvo
    imds.wvvVO = wvvVO
    imds.v1a = v1a
    imds.v1b = v1b
    imds.v2a = v2a
    imds.v2b = v2b
    imds.w3a = w3a
    imds.w3b = w3b
    return imds


# update l1, l2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    fova = eris.fov
    fovb = eris.fOV
    v1a = imds.v1a - eris._fvv
    v1b = imds.v1b - eris._fVV
    v2a = imds.v2a - eris._foo
    v2b = imds.v2b - eris._fOO

    mvv = einsum('klca,klcb->ba', l2aa, t2aa) * .5
    mvv+= einsum('lkac,lkbc->ba', l2ab, t2ab)
    mVV = einsum('klca,klcb->ba', l2bb, t2bb) * .5
    mVV+= einsum('klca,klcb->ba', l2ab, t2ab)
    moo = einsum('kicd,kjcd->ij', l2aa, t2aa) * .5
    moo+= einsum('ikdc,jkdc->ij', l2ab, t2ab)
    mOO = einsum('kicd,kjcd->ij', l2bb, t2bb) * .5
    mOO+= einsum('kicd,kjcd->ij', l2ab, t2ab)

    m3aa = einsum('ijcd,cadb->ijab', l2aa, eris.vvvv)
    m3ab = einsum('ijcd,cadb->ijab', l2ab, eris.vvVV)
    m3bb = einsum('ijcd,cadb->ijab', l2bb, eris.VVVV)

    m3aa += einsum('klab,ikjl->ijab', l2aa, imds.woooo)
    m3bb += einsum('klab,ikjl->ijab', l2bb, imds.wOOOO)
    m3ab += einsum('klab,ikjl->ijab', l2ab, imds.wooOO)

    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    ovOV = eris.ovOV
    mvv1 = einsum('jc,jb->bc', l1a, t1a) + mvv
    mVV1 = einsum('jc,jb->bc', l1b, t1b) + mVV
    moo1 = einsum('ic,kc->ik', l1a, t1a) + moo
    mOO1 = einsum('ic,kc->ik', l1b, t1b) + mOO

    ovvv = eris.vovv.transpose(1,0,3,2).conj() - eris.vovv.transpose(1,2,3,0).conj()
    tmp = einsum('ijcd,kd->ijck', l2aa, t1a)
    m3aa -= einsum('kbca,ijck->ijab', ovvv, tmp)

    tmp = einsum('ic,jbca->jiba', l1a, ovvv)
    tmp+= einsum('kiab,jk->ijab', l2aa, v2a)
    tmp-= einsum('ik,kajb->ijab', moo1, ovov)
    u2aa = tmp - tmp.transpose(1,0,2,3)
    u1a = einsum('iacb,bc->ia', ovvv, mvv1)
    ovvv = tmp = None

    OVVV = eris.VOVV.transpose(1,0,3,2).conj() - eris.VOVV.transpose(1,2,3,0).conj()
    tmp = einsum('ijcd,kd->ijck', l2bb, t1b)
    m3bb -= einsum('kbca,ijck->ijab', OVVV, tmp)

    tmp = einsum('ic,jbca->jiba', l1b, OVVV)
    tmp+= einsum('kiab,jk->ijab', l2bb, v2b)
    tmp-= einsum('ik,kajb->ijab', mOO1, OVOV)
    u2bb = tmp - tmp.transpose(1,0,2,3)
    u1b = einsum('iacb,bc->ia', OVVV, mVV1)
    OVVV = tmp = None

    OVvv = eris.VOvv.transpose(1,0,3,2)
    tmp = einsum('ijcd,kd->ijck', l2ab, t1b)
    m3ab -= einsum('kbca,ijck->ijab', OVvv, tmp)

    tmp = einsum('ic,jacb->jiba', l1a, OVvv)
    tmp-= einsum('kiab,jk->ijab', l2ab, v2a)
    tmp-= einsum('ik,jakb->ijab', mOO1, ovOV)
    u2ab = tmp.transpose(1,0,2,3)
    u1b += einsum('iacb,bc->ia', OVvv, mvv1)
    OVvv = tmp = None

    ovVV = eris.voVV.transpose(1,0,3,2)
    tmp = einsum('ijdc,kd->ijck', l2ab, t1a)
    m3ab -= einsum('kacb,ijck->ijab', ovVV, tmp)

    tmp = einsum('ic,jbca->jiba', l1b, ovVV)
    tmp-= einsum('ikab,jk->ijab', l2ab, v2b)
    tmp-= einsum('ik,kajb->ijab', moo1, ovOV)
    u2ab += tmp
    u1a += einsum('iacb,bc->ia', ovVV, mVV1)
    ovVV = tmp = None

    tauaa, tauab, taubb = make_tau(t2, t1, t1)
    tmp = einsum('ijcd,klcd->ijkl', l2aa, tauaa)
    ovov = eris.ovov - eris.ovov.transpose(0,3,2,1)
    m3aa += einsum('kalb,ijkl->ijab', ovov, tmp) * .25

    tmp = einsum('ijcd,klcd->ijkl', l2bb, taubb)
    OVOV = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
    m3bb += einsum('kalb,ijkl->ijab', OVOV, tmp) * .25

    tmp = einsum('ijcd,klcd->ijkl', l2ab, tauab)
    ovOV = eris.ovOV
    m3ab += einsum('kalb,ijkl->ijab', ovOV, tmp) * .5
    tmp = einsum('ijdc,lkdc->ijkl', l2ab, tauab)
    m3ab += einsum('lakb,ijkl->ijab', ovOV, tmp) * .5

    u1a += einsum('ijab,jb->ia', m3aa, t1a)
    u1a += einsum('ijab,jb->ia', m3ab, t1b)
    u1b += einsum('ijab,jb->ia', m3bb, t1b)
    u1b += einsum('jiba,jb->ia', m3ab, t1a)

    u2aa += m3aa
    u2bb += m3bb
    u2ab += m3ab
    u2aa += ovov.transpose(0,2,1,3)
    u2bb += OVOV.transpose(0,2,1,3)
    u2ab += ovOV.transpose(0,2,1,3)

    fov1 = fova + einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= einsum('jbkc,kc->jb', ovOV, t1b)
    tmp = einsum('ia,jb->ijab', l1a, fov1)
    tmp+= einsum('kica,jbck->ijab', l2aa, imds.wovvo)
    tmp+= einsum('ikac,jbck->ijab', l2ab, imds.wovVO)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2aa += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= einsum('kcjb,kc->jb', ovOV, t1a)
    tmp = einsum('ia,jb->ijab', l1b, fov1)
    tmp+= einsum('kica,jbck->ijab', l2bb, imds.wOVVO)
    tmp+= einsum('kica,jbck->ijab', l2ab, imds.wOVvo)
    tmp = tmp - tmp.transpose(1,0,2,3)
    u2bb += tmp - tmp.transpose(0,1,3,2)

    fov1 = fovb + einsum('kcjb,kc->jb', OVOV, t1b)
    fov1+= einsum('kcjb,kc->jb', ovOV, t1a)
    u2ab += einsum('ia,jb->ijab', l1a, fov1)
    u2ab += einsum('ikac,jbck->ijab', l2ab, imds.wOVVO)
    u2ab += einsum('kica,jbck->ijab', l2aa, imds.wOVvo)
    u2ab += einsum('kiac,jbck->jiab', l2ab, imds.woVVo)
    u2ab += einsum('ikca,jbck->ijba', l2ab, imds.wOvvO)
    fov1 = fova + einsum('kcjb,kc->jb', ovov, t1a)
    fov1+= einsum('jbkc,kc->jb', ovOV, t1b)
    u2ab += einsum('ia,jb->jiba', l1b, fov1)
    u2ab += einsum('kica,jbck->jiba', l2ab, imds.wovvo)
    u2ab += einsum('kica,jbck->jiba', l2bb, imds.wovVO)

    ovoo = eris.ooov.transpose(2,3,0,1) - eris.ooov.transpose(0,3,2,1)
    OVOO = eris.OOOV.transpose(2,3,0,1) - eris.OOOV.transpose(0,3,2,1)
    OVoo = eris.ooOV.transpose(2,3,0,1)
    ovOO = eris.OOov.transpose(2,3,0,1)
    tmp = einsum('ka,jbik->ijab', l1a, ovoo)
    tmp+= einsum('ijca,cb->ijab', l2aa, v1a)
    tmp+= einsum('ca,icjb->ijab', mvv1, ovov)
    u2aa -= tmp - tmp.transpose(0,1,3,2)
    tmp = einsum('ka,jbik->ijab', l1b, OVOO)
    tmp+= einsum('ijca,cb->ijab', l2bb, v1b)
    tmp+= einsum('ca,icjb->ijab', mVV1, OVOV)
    u2bb -= tmp - tmp.transpose(0,1,3,2)
    u2ab -= einsum('ka,jbik->ijab', l1a, OVoo)
    u2ab += einsum('ijac,cb->ijab', l2ab, v1b)
    u2ab -= einsum('ca,icjb->ijab', mvv1, ovOV)
    u2ab -= einsum('ka,ibjk->ijba', l1b, ovOO)
    u2ab += einsum('ijca,cb->ijba', l2ab, v1a)
    u2ab -= einsum('ca,ibjc->ijba', mVV1, ovOV)

    u1a += fova
    u1b += fovb
    u1a += einsum('ib,ba->ia', l1a, v1a)
    u1a -= einsum('ja,ij->ia', l1a, v2a)
    u1b += einsum('ib,ba->ia', l1b, v1b)
    u1b -= einsum('ja,ij->ia', l1b, v2b)

    u1a += einsum('jb,bjia->ia', l1a, eris.voov)
    u1a -= einsum('jb,ijba->ia', l1a, eris.oovv)
    u1a += einsum('jb,bjia->ia', l1b, eris.VOov)
    u1b += einsum('jb,bjia->ia', l1b, eris.VOOV)
    u1b -= einsum('jb,ijba->ia', l1b, eris.OOVV)
    u1b += einsum('jb,bjia->ia', l1a, eris.voOV)

    u1a -= einsum('kjca,ijck->ia', l2aa, imds.woovo)
    u1a -= einsum('jkac,ijck->ia', l2ab, imds.wooVO)
    u1b -= einsum('kjca,ijck->ia', l2bb, imds.wOOVO)
    u1b -= einsum('kjca,ijck->ia', l2ab, imds.wOOvo)

    u1a -= einsum('ikbc,back->ia', l2aa, imds.wvvvo)
    u1a -= einsum('ikbc,back->ia', l2ab, imds.wvvVO)
    u1b -= einsum('ikbc,back->ia', l2bb, imds.wVVVO)
    u1b -= einsum('kicb,back->ia', l2ab, imds.wVVvo)

    u1a += einsum('jiba,bj->ia', l2aa, imds.w3a)
    u1a += einsum('ijab,bj->ia', l2ab, imds.w3b)
    u1b += einsum('jiba,bj->ia', l2bb, imds.w3b)
    u1b += einsum('jiba,bj->ia', l2ab, imds.w3a)

    tmpa  = t1a + einsum('kc,kjcb->jb', l1a, t2aa)
    tmpa += einsum('kc,jkbc->jb', l1b, t2ab)
    tmpa -= einsum('bd,jd->jb', mvv1, t1a)
    tmpa -= einsum('lj,lb->jb', moo, t1a)
    tmpb  = t1b + einsum('kc,kjcb->jb', l1b, t2bb)
    tmpb += einsum('kc,kjcb->jb', l1a, t2ab)
    tmpb -= einsum('bd,jd->jb', mVV1, t1b)
    tmpb -= einsum('lj,lb->jb', mOO, t1b)
    u1a += einsum('jbia,jb->ia', ovov, tmpa)
    u1a += einsum('iajb,jb->ia', ovOV, tmpb)
    u1b += einsum('jbia,jb->ia', OVOV, tmpb)
    u1b += einsum('jbia,jb->ia', ovOV, tmpa)

    u1a -= einsum('iajk,kj->ia', ovoo, moo1)
    u1a -= einsum('iajk,kj->ia', ovOO, mOO1)
    u1b -= einsum('iajk,kj->ia', OVOO, mOO1)
    u1b -= einsum('iajk,kj->ia', OVoo, moo1)

    tmp  = fova - einsum('kbja,jb->ka', ovov, t1a)
    tmp += einsum('kajb,jb->ka', ovOV, t1b)
    u1a -= einsum('ik,ka->ia', moo, tmp)
    u1a -= einsum('ca,ic->ia', mvv, tmp)
    tmp  = fovb - einsum('kbja,jb->ka', OVOV, t1b)
    tmp += einsum('jbka,jb->ka', ovOV, t1a)
    u1b -= einsum('ik,ka->ia', mOO, tmp)
    u1b -= einsum('ca,ic->ia', mVV, tmp)

    u1a /= eris.eia
    u1b /= eris.eIA

    u2aa /= eris.eijab
    u2ab /= eris.eiJaB
    u2bb /= eris.eIJAB

    time0 = log.timer_debug1('update l1 l2', *time0)
    return (u1a,u1b), (u2aa,u2ab,u2bb)


class UCCSD(uccsd.UCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        uccsd.UCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        self.ip_partition = self.ea_partition = None
        self._keys = self._keys.union(['max_space', 'ip_partition', 'ea_partition'])

    def init_amps(self, eris=None):
        e, t1, t2 = init_amps(self, eris)
        self.emp2 = e.real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    energy = energy
    update_amps = update_amps

    def ao2mo(self, mo_coeff=None):
        return _make_eris_incore(self, mo_coeff)

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
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return uccsd_t_slow(self, eris, t1, t2)

    def ccsd_t(self, t1=None, t2=None, eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        raise NotImplementedError

    nip = eom_uccsd.EOMIP.vector_size
    nea = eom_uccsd.EOMEA.vector_size

    def vector_to_amplitudes_ip(self, vector, **kwargs):
        return eom_uccsd.EOMIP.vector_to_amplitudes(self, vector)

    def vector_to_amplitudes_ea(self, vector, **kwargs):
        return eom_uccsd.EOMEA.vector_to_amplitudes(self, vector)

    amplitudes_to_vector_ip = eom_uccsd.EOMIP.amplitudes_to_vector
    amplitudes_to_vector_ea = eom_uccsd.EOMEA.amplitudes_to_vector

    def ipccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        t1, t2 = imds.t1, imds.t2
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape

        Hr1a = -imds.Foo.diagonal()
        Hr1b = -imds.FOO.diagonal()

        Hr2aaa = imds.Fvv.diagonal().reshape(1,1,nvira) -\
                 imds.Foo.diagonal().reshape(nocca,1,1) -\
                 imds.Foo.diagonal().reshape(1,nocca,1)

        Hr2bbb = imds.FVV.diagonal().reshape(1,1,nvirb) -\
                 imds.FOO.diagonal().reshape(noccb,1,1) -\
                 imds.FOO.diagonal().reshape(1,noccb,1)

        Hr2abb = imds.FVV.diagonal().reshape(1,1,nvirb) -\
                 imds.Foo.diagonal().reshape(nocca,1,1) -\
                 imds.FOO.diagonal().reshape(1,noccb,1)

        Hr2baa = imds.Fvv.diagonal().reshape(1,1,nvira) -\
                 imds.FOO.diagonal().reshape(noccb,1,1) -\
                 imds.Foo.diagonal().reshape(1,nocca,1)


        if self.ip_partition != 'mp':
            Hr2aaa = Hr2aaa + einsum('iijj->ij', imds.Woooo).reshape(nocca,nocca,1)
            Hr2abb = Hr2abb + einsum('iijj->ij', imds.WooOO).reshape(nocca,noccb,1)
            Hr2bbb = Hr2bbb + einsum('iijj->ij', imds.WOOOO).reshape(noccb,noccb,1)
            Hr2baa = Hr2baa + einsum('jjii->ij', imds.WooOO).reshape(noccb,nocca,1)

            Hr2aaa -= einsum('iejb,jibe->ijb', imds.Wovov, t2aa)
            Hr2abb -= einsum('iejb,ijeb->ijb', imds.WovOV, t2ab)
            Hr2baa -= einsum('jbie,jibe->ijb', imds.WovOV, t2ab)
            Hr2bbb -= einsum('iejb,jibe->ijb', imds.WOVOV, t2bb)
            Hr2aaa = Hr2aaa + einsum('ibbi->ib', imds.Wovvo).reshape(nocca,1,nvira)
            Hr2aaa = Hr2aaa + einsum('jbbj->jb', imds.Wovvo).reshape(1,nocca,nvira)

            Hr2baa = Hr2baa + einsum('jbbj->jb', imds.Wovvo).reshape(1,nocca,nvira)
            Hr2baa = Hr2baa - einsum('iibb->ib', imds.WOOvv).reshape(noccb,1,nvira)

            Hr2abb = Hr2abb + einsum('jbbj->jb', imds.WOVVO).reshape(1,noccb,nvirb)
            Hr2abb = Hr2abb - einsum('iibb->ib', imds.WooVV).reshape(nocca,1,nvirb)

            Hr2bbb = Hr2bbb + einsum('ibbi->ib', imds.WOVVO).reshape(noccb,1,nvirb)
            Hr2bbb = Hr2bbb + einsum('jbbj->jb', imds.WOVVO).reshape(1,noccb,nvirb)

        vector = self.amplitudes_to_vector_ip((Hr1a,Hr1b), (Hr2aaa,Hr2baa,Hr2abb,Hr2bbb))
        return vector

    def eaccsd_diag(self):
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds
        t1, t2 = imds.t1, imds.t2
        t1a, t1b = t1
        t2aa, t2ab, t2bb = t2
        nocca, noccb, nvira, nvirb = t2ab.shape

        Hr1a = imds.Fvv.diagonal()
        Hr1b = imds.FVV.diagonal()

        Hr2aaa =-imds.Foo.diagonal().reshape(nocca,1,1) +\
                 imds.Fvv.diagonal().reshape(1,nvira,1) +\
                 imds.Fvv.diagonal().reshape(1,1,nvira)

        Hr2aba =-imds.Foo.diagonal().reshape(nocca,1,1) +\
                 imds.FVV.diagonal().reshape(1,nvirb,1) +\
                 imds.Fvv.diagonal().reshape(1,1,nvira)

        Hr2bab =-imds.FOO.diagonal().reshape(noccb,1,1) +\
                 imds.Fvv.diagonal().reshape(1,nvira,1) +\
                 imds.FVV.diagonal().reshape(1,1,nvirb)

        Hr2bbb =-imds.FOO.diagonal().reshape(noccb,1,1) +\
                 imds.FVV.diagonal().reshape(1,nvirb,1) +\
                 imds.FVV.diagonal().reshape(1,1,nvirb)

        if self.ea_partition != 'mp':
            Hr2aaa = Hr2aaa + einsum('aabb->ab', imds.Wvvvv).reshape(1,nvira,nvira)
            Hr2aba = Hr2aba + einsum('bbaa->ab', imds.WvvVV).reshape(1,nvirb,nvira)
            Hr2bab = Hr2bab + einsum('aabb->ab', imds.WvvVV).reshape(1,nvira,nvirb)
            Hr2bbb = Hr2bbb + einsum('aabb->ab', imds.WVVVV).reshape(1,nvirb,nvirb)

            # Wovov term (physicist's Woovv)
            Hr2aaa -= einsum('kajb,kjab->jab', imds.Wovov, t2aa)
            Hr2aba -= einsum('jbka,jkba->jab', imds.WovOV, t2ab)
            Hr2bab -= einsum('kajb,kjab->jab', imds.WovOV, t2ab)
            Hr2bbb -= einsum('kajb,kjab->jab', imds.WOVOV, t2bb)

        # Wovvo term
            Hr2aaa = Hr2aaa + einsum('jbbj->jb', imds.Wovvo).reshape(nocca,1,nvira)
            Hr2aaa = Hr2aaa + einsum('jaaj->ja', imds.Wovvo).reshape(nocca,nvira,1)

            Hr2aba = Hr2aba + einsum('jbbj->jb', imds.Wovvo).reshape(nocca,1,nvira)
            Hr2aba = Hr2aba - einsum('jjaa->ja', imds.WooVV).reshape(nocca,nvirb,1)

            Hr2bab = Hr2bab + einsum('jbbj->jb', imds.WOVVO).reshape(noccb,1,nvirb)
            Hr2bab = Hr2bab - einsum('jjaa->ja', imds.WOOvv).reshape(noccb,nvira,1)

            Hr2bbb = Hr2bbb + einsum('jbbj->jb', imds.WOVVO).reshape(noccb,1,nvirb)
            Hr2bbb = Hr2bbb + einsum('jaaj->ja', imds.WOVVO).reshape(noccb,nvirb,1)

        vector = self.amplitudes_to_vector_ea([Hr1a,Hr1b], [Hr2aaa,Hr2aba,Hr2bab,Hr2bbb])
        return vector

    def ipccsd_matvec(self, vector, **kwargs):
        # Ref: Tu, Wang, and Li, J. Chem. Phys. 136, 174102 (2012) Eqs.(8)-(9)
        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds

        r1, r2 = self.vector_to_amplitudes_ip(vector, **kwargs)

        t2aa, t2ab, t2bb = imds.t2

        r1a, r1b = r1
        r2aaa, r2baa, r2abb, r2bbb = r2

        Hr1a = -einsum('mi,m->i', imds.Foo, r1a)
        Hr1b = -einsum('mi,m->i', imds.FOO, r1b)

        Hr1a += einsum('me,mie->i', imds.Fov, r2aaa)
        Hr1a -= einsum('me,ime->i', imds.FOV, r2abb)
        Hr1b += einsum('me,mie->i', imds.FOV, r2bbb)
        Hr1b -= einsum('me,ime->i', imds.Fov, r2baa)


        Hr1a += -0.5 * einsum('nime,mne->i', imds.Wooov, r2aaa)
        Hr1b +=    einsum('nime,nme->i', imds.WOOov, r2baa)
        Hr1b += -0.5 * einsum('nime,mne->i', imds.WOOOV, r2bbb)
        Hr1a +=    einsum('nime,nme->i', imds.WooOV, r2abb)


        Hr2aaa = einsum('be,ije->ijb', imds.Fvv, r2aaa)
        Hr2abb = einsum('be,ije->ijb', imds.FVV, r2abb)
        Hr2bbb = einsum('be,ije->ijb', imds.FVV, r2bbb)
        Hr2baa = einsum('be,ije->ijb', imds.Fvv, r2baa)


        tmpa = einsum('mi,mjb->ijb', imds.Foo, r2aaa)
        tmpb = einsum('mj,mib->ijb', imds.Foo, r2aaa)
        Hr2aaa -= tmpa - tmpb
        Hr2abb -= einsum('mi,mjb->ijb', imds.Foo, r2abb)
        Hr2abb -= einsum('mj,imb->ijb', imds.FOO, r2abb)
        Hr2baa -= einsum('mi,mjb->ijb', imds.FOO, r2baa)
        Hr2baa -= einsum('mj,imb->ijb', imds.Foo, r2baa)
        tmpb = einsum('mi,mjb->ijb', imds.FOO, r2bbb)
        tmpa = einsum('mj,mib->ijb', imds.FOO, r2bbb)
        Hr2bbb -= tmpb - tmpa

        Hr2aaa -= einsum('mjbi,m->ijb', imds.Woovo, r1a)
        Hr2abb += einsum('mibj,m->ijb', imds.WooVO, r1a)
        Hr2baa += einsum('mibj,m->ijb', imds.WOOvo, r1b)
        Hr2bbb -= einsum('mjbi,m->ijb', imds.WOOVO, r1b)

        Hr2aaa += .5 * einsum('minj,mnb->ijb', imds.Woooo, r2aaa)
        Hr2abb +=      einsum('minj,mnb->ijb', imds.WooOO, r2abb)
        Hr2bbb += .5 * einsum('minj,mnb->ijb', imds.WOOOO, r2bbb)
        Hr2baa +=      einsum('njmi,mnb->ijb', imds.WooOO, r2baa)

        tmp_aaa = einsum('menf,mnf->e', imds.Wovov, r2aaa)
        tmp_bbb = einsum('menf,mnf->e', imds.WOVOV, r2bbb)
        tmp_abb = einsum('menf,mnf->e', imds.WovOV, r2abb)
        tmp_baa = einsum('nfme,mnf->e', imds.WovOV, r2baa)

        Hr2aaa -= 0.5 * einsum('e,jibe->ijb', tmp_aaa, t2aa)
        Hr2aaa -= einsum('e,jibe->ijb', tmp_abb, t2aa)

        Hr2abb -= 0.5 * einsum('e,ijeb->ijb', tmp_aaa, t2ab)
        Hr2abb -= einsum('e,ijeb->ijb', tmp_abb, t2ab)

        Hr2baa -= 0.5 * einsum('e,jibe->ijb', tmp_bbb, t2ab)
        Hr2baa -= einsum('e,jibe->ijb', tmp_baa, t2ab)

        Hr2bbb -= 0.5 * einsum('e,jibe->ijb', tmp_bbb, t2bb)
        Hr2bbb -= einsum('e,jibe->ijb', tmp_baa, t2bb)

        Hr2aaa += einsum('mebj,ime->ijb', imds.Wovvo, r2aaa)
        Hr2aaa += einsum('mebj,ime->ijb', imds.WOVvo, r2abb)
        # P(ij)
        Hr2aaa -= einsum('mebi,jme->ijb', imds.Wovvo, r2aaa)
        Hr2aaa -= einsum('mebi,jme->ijb', imds.WOVvo, r2abb)

        Hr2abb += einsum('mebj,ime->ijb', imds.WovVO, r2aaa)
        Hr2abb += einsum('mebj,ime->ijb', imds.WOVVO, r2abb)
        Hr2abb -= einsum('mibe,mje->ijb', imds.WooVV, r2abb)

        Hr2baa += einsum('mebj,ime->ijb', imds.WOVvo, r2bbb)
        Hr2baa += einsum('mebj,ime->ijb', imds.Wovvo, r2baa)
        Hr2baa -= einsum('mibe,mje->ijb', imds.WOOvv, r2baa)


        Hr2bbb += einsum('mebj,ime->ijb', imds.WOVVO, r2bbb)
        Hr2bbb += einsum('mebj,ime->ijb', imds.WovVO, r2baa)
        # P(ij)
        Hr2bbb -= einsum('mebi,jme->ijb', imds.WOVVO, r2bbb)
        Hr2bbb -= einsum('mebi,jme->ijb', imds.WovVO, r2baa)
        vector = self.amplitudes_to_vector_ip([Hr1a, Hr1b], [Hr2aaa, Hr2baa, Hr2abb, Hr2bbb])
        return vector

    def eaccsd_matvec(self, vector, **kwargs):

        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds
        t2aa, t2ab, t2bb = imds.t2
        r1, r2 = self.vector_to_amplitudes_ea(vector, **kwargs)
        r1a, r1b = r1
        r2aaa, r2aba, r2bab, r2bbb = r2

        Hr1a = einsum('ac,c->a', imds.Fvv, r1a)
        Hr1b = einsum('ac,c->a', imds.FVV, r1b)

        Hr1a += einsum('ld,lad->a', imds.Fov, r2aaa)
        Hr1a += einsum('ld,lad->a', imds.FOV, r2bab)
        Hr1b += einsum('ld,lad->a', imds.Fov, r2aba)
        Hr1b += einsum('ld,lad->a', imds.FOV, r2bbb)

        Hr1a += 0.5*einsum('acld,lcd->a', imds.Wvvov, r2aaa)
        Hr1a +=     einsum('acld,lcd->a', imds.WvvOV, r2bab)
        Hr1b += 0.5*einsum('acld,lcd->a', imds.WVVOV, r2bbb)
        Hr1b +=     einsum('acld,lcd->a', imds.WVVov, r2aba)

        Hr2aaa = .5 * einsum('acbd,jcd->jab', imds.Wvvvv, r2aaa)
        Hr2aba =      einsum('bcad,jdc->jab', imds.WvvVV, r2aba)
        Hr2bab =      einsum('acbd,jcd->jab', imds.WvvVV, r2bab)
        Hr2bbb = .5 * einsum('acbd,jcd->jab', imds.WVVVV, r2bbb)

        Hr2aaa += einsum('acbj,c->jab', imds.Wvvvo, r1a)
        Hr2bbb += einsum('acbj,c->jab', imds.WVVVO, r1b)

        Hr2bab += einsum('acbj,c->jab', imds.WvvVO, r1a)
        Hr2aba += einsum('acbj,c->jab', imds.WVVvo, r1b)

        tmpa = einsum('ac,jcb->jab', imds.Fvv, r2aaa)
        tmpb = einsum('bc,jca->jab', imds.Fvv, r2aaa)
        Hr2aaa += tmpa - tmpb
        Hr2aba += einsum('ac,jcb->jab', imds.FVV, r2aba)
        Hr2bab += einsum('ac,jcb->jab', imds.Fvv, r2bab)
        Hr2aba += einsum('bc,jac->jab', imds.Fvv, r2aba)
        Hr2bab += einsum('bc,jac->jab', imds.FVV, r2bab)
        tmpb = einsum('ac,jcb->jab', imds.FVV, r2bbb)
        tmpa = einsum('bc,jca->jab', imds.FVV, r2bbb)
        Hr2bbb += tmpb - tmpa

        Hr2aaa -= einsum('lj,lab->jab', imds.Foo, r2aaa)
        Hr2bbb -= einsum('lj,lab->jab', imds.FOO, r2bbb)
        Hr2bab -= einsum('lj,lab->jab', imds.FOO, r2bab)
        Hr2aba -= einsum('lj,lab->jab', imds.Foo, r2aba)

        tmp_aaa = einsum('kcld,lcd->k', imds.Wovov, r2aaa)
        tmp_bbb = einsum('kcld,lcd->k', imds.WOVOV, r2bbb)
        tmp_bab = einsum('kcld,lcd->k', imds.WovOV, r2bab)
        tmp_aba = einsum('ldkc,lcd->k', imds.WovOV, r2aba)

        Hr2aaa -= 0.5 * einsum('k,kjab->jab', tmp_aaa, t2aa)
        Hr2bab -= 0.5 * einsum('k,kjab->jab', tmp_aaa, t2ab)

        Hr2aaa -= einsum('k,kjab->jab', tmp_bab, t2aa)
        Hr2bbb -= 0.5 * einsum('k,kjab->jab', tmp_bbb, t2bb)

        Hr2bbb -= einsum('k,kjab->jab', tmp_aba, t2bb)
        Hr2bab -= einsum('k,kjab->jab', tmp_bab, t2ab)

        Hr2aba -= einsum('k,jkba->jab', tmp_aba, t2ab)
        Hr2aba -= 0.5 * einsum('k,jkba->jab', tmp_bbb, t2ab)

        Hr2aaa += einsum('ldbj,lad->jab', imds.Wovvo, r2aaa)
        Hr2aaa += einsum('ldbj,lad->jab', imds.WOVvo, r2bab)
        # P(ab)
        Hr2aaa -= einsum('ldaj,lbd->jab', imds.Wovvo, r2aaa)
        Hr2aaa -= einsum('ldaj,lbd->jab', imds.WOVvo, r2bab)

        Hr2bab += einsum('ldbj,lad->jab', imds.WovVO, r2aaa)
        Hr2bab += einsum('ldbj,lad->jab', imds.WOVVO, r2bab)
        Hr2bab -= einsum('ljad,ldb->jab', imds.WOOvv, r2bab)

        Hr2aba += einsum('ldbj,lad->jab', imds.WOVvo, r2bbb)
        Hr2aba += einsum('ldbj,lad->jab', imds.Wovvo, r2aba)
        Hr2aba -= einsum('ljad,ldb->jab', imds.WooVV, r2aba)

        Hr2bbb += einsum('ldbj,lad->jab', imds.WOVVO, r2bbb)
        Hr2bbb += einsum('ldbj,lad->jab', imds.WovVO, r2aba)
        # P(ab)
        Hr2bbb -= einsum('ldaj,lbd->jab', imds.WOVVO, r2bbb)
        Hr2bbb -= einsum('ldaj,lbd->jab', imds.WovVO, r2aba)

        vector = self.amplitudes_to_vector_ea([Hr1a, Hr1b], [Hr2aaa, Hr2aba, Hr2bab, Hr2bbb])
        return vector

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

        if not getattr(self, 'imds', None):
            self.imds = _IMDS(self)
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        if partition == 'full':
            self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(self.ipccsd_diag())[1]

        adiag = self.ipccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            if koopmans:
                nocca, noccb = self.nocc
                idx = adiag[:nocca+noccb].argsort()
            else:
                idx = adiag.argsort()
            dtype = getattr(adiag, 'dtype', np.double)
            guess = []
            for i in idx[:nroots]:
                g = np.zeros(size, dtype)
                g[i] = 1.0
                guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        real_system = (self._scf.mo_coeff[0].dtype == np.double)
        if left:
            matvec = lambda xs: [self.lipccsd_matvec(x) for x in xs]
        else:
            matvec = lambda xs: [self.ipccsd_matvec(x) for x in xs]

        eig = lib.davidson_nosym1
        if user_guess or koopmans:
            assert len(guess) == nroots
            def eig_close_to_init_guess(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
            conv, eip, evecs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                               tol=self.conv_tol, max_cycle=self.max_cycle,
                               max_space=self.max_space, nroots=nroots, verbose=log)

        else:
            def pickeig(w, v, nroots, envs):
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
            conv, eip, evecs = eig(matvec, guess, precond, pick=pickeig,
                               tol=self.conv_tol, max_cycle=self.max_cycle,
                               max_space=self.max_space, nroots=nroots, verbose=log)


        self.eip = eip.real
        nocc = sum(self.nocc)
        if nroots == 1:
            eip, evecs = [self.eip], [evecs]
        for n, en, vn in zip(range(nroots), eip, evecs):
            logger.info(self, 'IP root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:nocc])**2)
        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        '''Calculate (N+1)-electron charged excitations via EA-EOM-CCSD.

        Kwargs:
            See ipccd()
        '''
        cput0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)
        size = self.nea()
        nroots = min(nroots,size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        if partition == 'full':
            self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(self.eaccsd_diag())[1]

        adiag = self.eaccsd_diag()
        user_guess = False
        if guess:
            user_guess = True
            assert len(guess) == nroots
            for g in guess:
                assert g.size == size
        else:
            if koopmans:
                nocca, noccb = self.nocc
                nmoa, nmob = self.nmo
                nvira, nvirb = nmoa-nocca, nmob-noccb
                idx = adiag[:nvira+nvirb].argsort()
            else:
                idx = adiag.argsort()

            dtype = getattr(adiag, 'dtype', np.double)
            nroots = min(nroots, size)
            guess = []
            for i in idx[:nroots]:
                g = np.zeros(size, dtype)
                g[i] = 1.0
                guess.append(g)

        def precond(r, e0, x0):
            return r/(e0-adiag+1e-12)

        real_system = (self._scf.mo_coeff[0].dtype == np.double)
        if left:
            matvec = lambda xs: [self.leaccsd_matvec(x) for x in xs]
        else:
            matvec = lambda xs: [self.eaccsd_matvec(x) for x in xs]

        eig = lib.davidson_nosym1
        if user_guess or koopmans:
            assert len(guess) == nroots
            def eig_close_to_init_guess(w, v, nroots, envs):
                x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
                s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
                snorm = np.einsum('pi,pi->i', s.conj(), s)
                idx = np.argsort(-snorm)[:nroots]
                return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
            conv, eea, evecs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                               tol=self.conv_tol, max_cycle=self.max_cycle,
                               max_space=self.max_space, nroots=nroots, verbose=log)

        else:
            def pickeig(w, v, nroots, envs):
                real_idx = np.where(abs(w.imag) < 1e-3)[0]
                return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
            conv, eea, evecs = eig(matvec, guess, precond, pick=pickeig,
                               tol=self.conv_tol, max_cycle=self.max_cycle,
                               max_space=self.max_space, nroots=nroots, verbose=log)

        self.eea = eea.real

        if nroots == 1:
            eea, evecs = [self.eea], [evecs]

        nvir = sum(self.nmo) - sum(self.nocc)
        for n, en, vn in zip(range(nroots), eea, evecs):
            logger.info(self, 'EA root %d E = %.16g  qpwt = %0.6g',
                        n, en, np.linalg.norm(vn[:nvir])**2)
        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomee_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

CCSD = UCCSD

class _ChemistsERIs(ccsd._ChemistsERIs):
    def __init__(self, mol=None):

        self.mol = mol
        self.mo_coeff = self.mo_energy =None
        self.nocc = None
        self.foo = self.fOO = self._foo = self._fOO = None
        self.fov = self.fOV = None
        self.fvv = self.fVV = self._fvv = self._fVV = None

        self.e_hf = None

        self.eia = None
        self.eIA = None

        self.eijab = None
        self.eiJaB = None
        self.eIJAB = None

        self.oooo = None
        self.ooov = None
        self.ovov = None
        self.voov = None
        self.oovv = None
        self.vovv = None
        self.vvvv = None

        self.OOOO = None
        self.OOOV = None
        self.OVOV = None
        self.VOOV = None
        self.OOVV = None
        self.VOVV = None
        self.VVVV = None

        self.ooOO = None
        self.ooOV = None
        self.ovOV = None
        self.voOV = None
        self.ooVV = None
        self.voVV = None
        self.vvVV = None

        self.OOov = None
        self.OOvv = None
        self.OVov = None
        self.VOov = None
        self.VOvv = None

    _common_init_ = uccsd._ChemistsERIs._common_init_

    def get_eijkabc(self, partition):
        nocca, noccb, nvira, nvirb = self.eiJaB.shape
        if partition == 'aaa':
            d3 = self.eia.reshape(nocca,1,1,nvira,1,1) + \
                 self.eia.reshape(1,nocca,1,1,nvira,1) + \
                 self.eia.reshape(1,1,nocca,1,1,nvira)
        elif partition == 'bbb':
            d3 = self.eIA.reshape(noccb,1,1,nvirb,1,1) + \
                 self.eIA.reshape(1,noccb,1,1,nvirb,1) + \
                 self.eIA.reshape(1,1,noccb,1,1,nvirb)
        elif partition == 'baa':
            d3 = self.eIA.reshape(noccb,1,1,nvirb,1,1) + \
                 self.eia.reshape(1,nocca,1,1,nvira,1) + \
                 self.eia.reshape(1,1,nocca,1,1,nvira)
        elif partition == 'abb':
            d3 = self.eia.reshape(nocca,1,1,nvira,1,1) + \
                 self.eIA.reshape(1,noccb,1,1,nvirb,1) + \
                 self.eIA.reshape(1,1,noccb,1,1,nvirb)
        else:
            raise ValueError("partition %s not recognized"%partition)
        return d3

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    eris.foo = eris.focka[:nocca,:nocca]
    eris.fov = eris.focka[:nocca,nocca:]
    eris.fvv = eris.focka[nocca:,nocca:]

    eris.fOO = eris.fockb[:noccb,:noccb]
    eris.fOV = eris.fockb[:noccb,noccb:]
    eris.fVV = eris.fockb[noccb:,noccb:]

    eris._foo = np.diag(np.diag(eris.foo))
    eris._fOO = np.diag(np.diag(eris.fOO))
    eris._fvv = np.diag(np.diag(eris.fvv))
    eris._fVV = np.diag(np.diag(eris.fVV))

    mo_ea, mo_eb = eris.mo_energy[0].real, eris.mo_energy[1].real
    eris.eia = mo_ea[:nocca][:,None] - mo_ea[nocca:][None,:]
    eris.eIA = mo_eb[:noccb][:,None] - mo_eb[noccb:][None,:]
    eris.eijab = eris.eia.reshape(nocca,1,nvira,1) + eris.eia.reshape(1,nocca,1,nvira)
    eris.eiJaB = eris.eia.reshape(nocca,1,nvira,1) + eris.eIA.reshape(1,noccb,1,nvirb)
    eris.eIJAB = eris.eIA.reshape(noccb,1,nvirb,1) + eris.eIA.reshape(1,noccb,1,nvirb)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, moa), nmoa)
        eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri, mob), nmob)
        eri_ab = ao2mo.general(mycc._scf._eri, (moa,moa,mob,mob), compact=False)
    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)

    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ooov = eri_aa[:nocca,:nocca,:nocca,nocca:].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.voov = eri_aa[nocca:,:nocca,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.vovv = eri_aa[nocca:,:nocca,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OOOV = eri_bb[:noccb,:noccb,:noccb,noccb:].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.VOOV = eri_bb[noccb:,:noccb,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.VOVV = eri_bb[noccb:,:noccb,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ooOV = eri_ab[:nocca,:nocca,:noccb,noccb:].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.voOV = eri_ab[nocca:,:nocca,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.voVV = eri_ab[nocca:,:nocca,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    eris.OOov = eri_ba[:noccb,:noccb,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.VOov = eri_ba[noccb:,:noccb,:nocca,nocca:].copy()
    eris.VOvv = eri_ba[noccb:,:noccb,nocca:,nocca:].copy()

    return eris

class _IMDS:
    def __init__(self, mycc, eris=None, t1=None, t2=None):
        self._cc = mycc
        self.verbose = mycc.verbose
        self.stdout = mycc.stdout
        if t1 is None:
            t1 = mycc.t1
        self.t1 = t1
        if t2 is None:
            t2 = mycc.t2
        self.t2 = t2
        if eris is None:
            if getattr(mycc, 'eris', None) is None:
                eris = mycc.ao2mo()
            else:
                eris = mycc.eris
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo, self.FOO = imd.Foo(t1, t2, eris)
        self.Fvv, self.FVV = imd.Fvv(t1, t2, eris)
        self.Fov, self.FOV = imd.Fov(t1, t2, eris)
        # 2 virtuals
        self.Wovvo, self.WovVO, self.WOVvo, self.WOVVO = imd.Wovvo(t1, t2, eris)
        self.Woovv, self.WooVV, self.WOOvv, self.WOOVV = imd.Woovv(t1, t2, eris)
        self.Wovov = eris.ovov - eris.ovov.transpose(2,1,0,3)
        self.WOVOV = eris.OVOV - eris.OVOV.transpose(2,1,0,3)
        self.WovOV = eris.ovOV.copy()
        self.WOVov = None
        self._made_shared = True
        logger.timer_debug1(self, 'EOM-UCCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        # 0 or 1 virtuals
        self.Woooo, self.WooOO, _         , self.WOOOO = imd.Woooo(t1, t2, eris)
        self.Wooov, self.WooOV, self.WOOov, self.WOOOV = imd.Wooov(t1, t2, eris)
        self.Woovo, self.WooVO, self.WOOvo, self.WOOVO = imd.Woovo(t1, t2, eris)
        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-UCCSD IP intermediates', *cput0)
        return self

    def make_ea(self):
        if not self._made_shared:
            self._make_shared()
        cput0 = (time.clock(), time.time())
        t1, t2, eris = self.t1, self.t2, self.eris
        # 3 or 4 virtuals
        self.Wvvov, self.WvvOV, self.WVVov, self.WVVOV = imd.Wvvov(t1, t2, eris)
        self.Wvvvv, self.WvvVV, self.WVVVV = imd.Wvvvv(t1, t2, eris)
        self.Wvvvo, self.WvvVO, self.WVVvo, self.WVVVO = imd.Wvvvo(t1, t2, eris)
        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-UCCSD EA intermediates', *cput0)
        return self

    def make_ee(self):
        raise NotImplementedError

if __name__ == '__main__':
    from pyscf import gto, cc

    mol = gto.Mole()
    mol.atom = [['O', (0.,   0., 0.)],
                ['O', (1.21, 0., 0.)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 2
    mol.build()
    mf = scf.UHF(mol).run()
    # Freeze 1s electrons
    # also acceptable
    #frozen = 4 or [2,2]
    frozen = [[0,1], [0,1]]
    ucc = UCCSD(mf, frozen=frozen)
    eris = ucc.ao2mo()
    ecc, t1, t2 = ucc.kernel(eris=eris)
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

    mycc = UCCSD(mf)
    mycc.direct = True
    ecc, t1, t2 = mycc.kernel()
    print(ecc - -0.2133432712431435)
    print(mycc.ccsd_t_slow() - -0.003060021865720902)

    e,v = mycc.ipccsd(nroots=8)
    print(e[0] - 0.4335604332073799)
    print(e[2] - 0.5187659896045407)
    print(e[4] - 0.6782876002229172)

    e,v = mycc.eaccsd(nroots=8)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)
    print(e[4] - 0.51006797826488071)
