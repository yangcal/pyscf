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

'''
Intermediates for restricted CCSD.  Complex integrals are supported.
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger

einsum = lib.einsum

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    Fki  = 2*einsum('kcld,ilcd->ki', eris.ovov, t2)
    Fki -=   einsum('kdlc,ilcd->ki', eris.ovov, t2)
    Fki += 2*einsum('kcld,ic,ld->ki', eris.ovov, t1, t1)
    Fki -=   einsum('kdlc,ic,ld->ki', eris.ovov, t1, t1)
    Fki += eris.foo
    return Fki

def cc_Fvv(t1, t2, eris):
    Fac  =-2*einsum('kcld,klad->ac', eris.ovov, t2)
    Fac +=   einsum('kdlc,klad->ac', eris.ovov, t2)
    Fac -= 2*einsum('kcld,ka,ld->ac', eris.ovov, t1, t1)
    Fac +=   einsum('kdlc,ka,ld->ac', eris.ovov, t1, t1)
    Fac += eris.fvv
    return Fac

def cc_Fov(t1, t2, eris):
    Fkc  = 2*einsum('kcld,ld->kc', eris.ovov, t1)
    Fkc -=   einsum('kdlc,ld->kc', eris.ovov, t1)
    Fkc += eris.fov
    return Fkc

### Eqs. (40)-(41) "lambda"

def Loo(t1, t2, eris):
    Lki = cc_Foo(t1, t2, eris) + einsum('kc,ic->ki', eris.fov, t1)
    Lki += 2*einsum('kilc,lc->ki', eris.ooov, t1)
    Lki -=   einsum('likc,lc->ki', eris.ooov, t1)
    return Lki

def Lvv(t1, t2, eris):
    Lac = cc_Fvv(t1, t2, eris) - einsum('kc,ka->ac', eris.fov, t1)
    Lac += 2*einsum('kdac,kd->ac', eris.ovvv, t1)
    Lac -=   einsum('kcad,kd->ac', eris.ovvv, t1)
    return Lac

### Eqs. (42)-(45) "chi"

def cc_Woooo(t1, t2, eris):
    Wklij  = einsum('kilc,jc->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    Wklij += einsum('kcld,ijcd->klij', eris.ovov, t2)
    Wklij += einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    # Incore
    Wabcd  = einsum('kdac,kb->abcd', eris.ovvv,-t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris.ovvv, t1)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    return Wabcd

def cc_Wvoov(t1, t2, eris):
    Wakic  = einsum('kcad,id->akic', eris.ovvv, t1)
    Wakic -= einsum('likc,la->akic', eris.ooov, t1)
    Wakic += eris.ovvo.transpose(2,0,3,1)
    Wakic -= 0.5*einsum('ldkc,ilda->akic', eris.ovov, t2)
    Wakic -= 0.5*einsum('lckd,ilad->akic', eris.ovov, t2)
    Wakic -= einsum('ldkc,id,la->akic', eris.ovov, t1, t1)
    Wakic += einsum('ldkc,ilad->akic', eris.ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):
    Wakci  = einsum('kdac,id->akci', eris.ovvv, t1)
    Wakci -= einsum('kilc,la->akci', eris.ooov, t1)
    Wakci += eris.oovv.transpose(2,0,3,1)
    Wakci -= 0.5*einsum('lckd,ilda->akci', eris.ovov, t2)
    Wakci -= einsum('lckd,id,la->akci', eris.ovov, t1, t1)
    return Wakci

def Wooov(t1, t2, eris):
    Wklid  = einsum('ic,kcld->klid', t1, eris.ovov)
    Wklid += eris.ooov.transpose(0,2,1,3)
    return Wklid

def Wvovv(t1, t2, eris):
    Walcd  = einsum('ka,kcld->alcd',-t1, eris.ovov)
    Walcd += eris.ovvv.transpose(2,0,3,1)
    return Walcd

def W1ovvo(t1, t2, eris):
    Wkaci  = 2*einsum('kcld,ilad->kaci', eris.ovov, t2)
    Wkaci +=  -einsum('kcld,liad->kaci', eris.ovov, t2)
    Wkaci +=  -einsum('kdlc,ilad->kaci', eris.ovov, t2)
    Wkaci += eris.ovvo.transpose(0,2,1,3)
    return Wkaci

def W2ovvo(t1, t2, eris):
    Wkaci = einsum('la,lkic->kaci',-t1, Wooov(t1, t2, eris))
    Wkaci += einsum('kcad,id->kaci', eris.ovvv, t1)
    return Wkaci

def Wovvo(t1, t2, eris):
    Wkaci = W1ovvo(t1, t2, eris) + W2ovvo(t1, t2, eris)
    return Wkaci

def W1ovov(t1, t2, eris):
    Wkbid = -einsum('kcld,ilcb->kbid', eris.ovov, t2)
    Wkbid += eris.oovv.transpose(0,2,1,3)
    return Wkbid

def W2ovov(t1, t2, eris):
    Wkbid = einsum('klid,lb->kbid', Wooov(t1, t2, eris),-t1)
    Wkbid += einsum('kcbd,ic->kbid', eris.ovvv, t1)
    return Wkbid

def Wovov(t1, t2, eris):
    return W1ovov(t1, t2, eris) + W2ovov(t1, t2, eris)

def Woooo(t1, t2, eris):
    Wklij  = einsum('kcld,ijcd->klij', eris.ovov, t2)
    Wklij += einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
    Wklij += einsum('kild,jd->klij', eris.ooov, t1)
    Wklij += einsum('ljkc,ic->klij', eris.ooov, t1)
    Wklij += eris.oooo.transpose(0,2,1,3)
    return Wklij

def Wvvvv(t1, t2, eris):
    Wabcd  = einsum('kcld,klab->abcd', eris.ovov, t2)
    Wabcd += einsum('kcld,ka,lb->abcd', eris.ovov, t1, t1)
    Wabcd += eris.vvvv.transpose(0,2,1,3)
    Wabcd -= einsum('ldac,lb->abcd', eris.ovvv, t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris.ovvv, t1)
    return Wabcd

def Wvvvo(t1, t2, eris, _Wvvvv=None):
    # Check if t1=0 (HF+MBPT(2))
    # don't make vvvv if you can avoid it!
    Wabcj  =  -einsum('alcj,lb->abcj', W1ovov(t1, t2, eris).transpose(1,0,3,2), t1)
    Wabcj +=  -einsum('kbcj,ka->abcj', W1ovvo(t1, t2, eris), t1)
    Wabcj += 2*einsum('ldac,ljdb->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('ldac,ljbd->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('lcad,ljdb->abcj', eris.ovvv, t2)
    Wabcj +=  -einsum('kcbd,jkda->abcj', eris.ovvv, t2)

    Wabcj +=   einsum('ljkc,lkba->abcj', eris.ooov, t2)
    Wabcj +=   einsum('ljkc,lb,ka->abcj', eris.ooov, t1, t1)
    Wabcj +=  -einsum('kc,kjab->abcj', cc_Fov(t1, t2, eris), t2)
    Wabcj += eris.ovvv.transpose(3,1,2,0).conj()
    if _Wvvvv is None:
        _Wvvvv = Wvvvv(t1, t2, eris)
    Wabcj += einsum('abcd,jd->abcj', _Wvvvv, t1)
    return Wabcj

def Wovoo(t1, t2, eris):
    Wkbij  =   einsum('kbid,jd->kbij', W1ovov(t1, t2, eris), t1)
    Wkbij +=  -einsum('klij,lb->kbij', Woooo(t1, t2, eris), t1)
    Wkbij +=   einsum('kbcj,ic->kbij', W1ovvo(t1, t2, eris), t1)
    Wkbij += 2*einsum('kild,ljdb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('kild,jldb->kbij', eris.ooov, t2)
    Wkbij +=  -einsum('likd,ljdb->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kcbd,jidc->kbij', eris.ovvv, t2)
    Wkbij +=   einsum('kcbd,jd,ic->kbij', eris.ovvv, t1, t1)
    Wkbij +=  -einsum('ljkc,libc->kbij', eris.ooov, t2)
    Wkbij +=   einsum('kc,ijcb->kbij', cc_Fov(t1, t2, eris), t2)
    Wkbij += eris.ooov.transpose(1,3,0,2).conj()
    return Wkbij

def get_t3p2_imds_slow(cc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    """Calculates T1, T2 amplitudes corrected by second-order T3 contribution
    and intermediates used in IP/EA-CCSD(T)a

    For description of arguments, see `get_t3p2_imds_slow` in `gintermediates.py`.
    """
    if eris is None:
        eris = cc.ao2mo()
    nocc, nvir = t1.shape

    fov = eris.fov

    dtype = np.result_type(t1.dtype, t2.dtype)
    if np.issubdtype(dtype, np.dtype(complex).type):
        logger.error(cc, 't3p2 imds has not been strictly checked for use with complex integrals')

    ovov = eris.ovov
    ovvv = eris.ovvv
    eris_vvov = eris.ovvv.conj().transpose(1,3,0,2)  # Physicist notation
    eris_vooo = eris.ooov.conj().transpose(3,2,1,0)  # Chemist notation

    ccsd_energy = cc.energy(t1, t2, eris)

    tmp_t3  = einsum('abif,kjcf->ijkabc', eris_vvov, t2)
    tmp_t3 -= einsum('aimj,mkbc->ijkabc', eris_vooo, t2)

    tmp_t3 = (tmp_t3 + tmp_t3.transpose(0,2,1,3,5,4)
                     + tmp_t3.transpose(1,0,2,4,3,5)
                     + tmp_t3.transpose(1,2,0,4,5,3)
                     + tmp_t3.transpose(2,0,1,5,3,4)
                     + tmp_t3.transpose(2,1,0,5,4,3))

    eia = eris.eia
    eijab = eris.eijab
    eijkabc = eris.get_eijkabc()
    tmp_t3 /= eijkabc

    Ptmp_t3 = 2.*tmp_t3 - tmp_t3.transpose(1,0,2,3,4,5) - tmp_t3.transpose(2,1,0,3,4,5)
    pt1 =  0.5 * einsum('jbkc,ijkabc->ia', 2.*ovov - ovov.transpose(0,3,2,1), Ptmp_t3)

    tmp =  0.5 * einsum('ijkabc,ia->jkbc', Ptmp_t3, fov)
    pt2 = tmp + tmp.transpose(1,0,3,2)

    #     b\    / \    /
    #  /\---\  /   \  /
    # i\/a  d\/j   k\/c
    # ------------------
    tmp =  einsum('ijkabc,iadb->jkdc', Ptmp_t3, ovvv)
    pt2 += (tmp + tmp.transpose(1,0,3,2))

    #     m\    / \    /
    #  /\---\  /   \  /
    # i\/a  j\/b   k\/c
    # ------------------
    tmp =  einsum('ijkabc,jmia->mkbc', Ptmp_t3, eris.ooov)
    pt2 -= (tmp + tmp.transpose(1,0,3,2))

    pt1 /= eia
    pt2 /= eijab

    pt1 += t1
    pt2 += t2

    Wmbkj =  einsum('ijkabc,mcia->mbkj', Ptmp_t3, ovov)

    Wcbej = -1.0*einsum('ijkabc,iake->cbej', Ptmp_t3, ovov)

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %14.8e', delta_ccsd_energy)
    return delta_ccsd_energy, pt1, pt2, Wmbkj, Wcbej
