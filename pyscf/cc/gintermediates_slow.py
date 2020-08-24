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

from pyscf import lib
import numpy as np
from pyscf.lib import logger

einsum = lib.einsum

# Ref: Gauss and Stanton, J. Chem. Phys. 103, 3561 (1995) Table III

# Section (a)

def make_tau(t2, t1a, t1b, eris, fac=1):
    t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    tau1 += t2
    return tau1

def cc_Fvv(t1, t2, eris):
    tau_tilde = make_tau(t2, t1, t1, eris, fac=0.5)
    Fae = eris.fvv - 0.5*einsum('me,ma->ae',eris.fov, t1)
    Fae += einsum('mf,mafe->ae', t1, eris.ovvv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo(t1, t2, eris):
    tau_tilde = make_tau(t2, t1, t1, eris, fac=0.5)
    Fmi = ( eris.foo + 0.5*einsum('me,ie->mi', eris.fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Fov(t1, t2, eris):
    Fme = eris.fov + einsum('nf,mnef->me', t1, eris.oovv)
    return Fme

def cc_Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1, eris)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1, eris)
    tmp = einsum('mb,mafe->bafe', t1, eris.ovvv)
    Wabef = eris.vvvv - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau, 0.25*eris.oovv)
    return Wabef

def cc_Wovvo(t1, t2, eris):
    eris_ovvo = -eris.ovov.transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej += einsum('nb,mnje->mbej', t1, eris.ooov)

    Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    tmp = einsum('nb,mnef->mbef', t1, eris.oovv)
    Wmbej -= einsum('jf,mbef->mbej', t1, tmp)
    Wmbej += eris_ovvo
    return Wmbej

### Section (b)

def Fvv(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fae = cc_Fvv(t1, t2, eris) - 0.5*einsum('ma,me->ae', t1,ccFov)
    return Fae

def Foo(t1, t2, eris):
    ccFov = cc_Fov(t1, t2, eris)
    Fmi = cc_Foo(t1, t2, eris) + 0.5*einsum('ie,me->mi', t1,ccFov)
    return Fmi

def Fov(t1, t2, eris):
    Fme = cc_Fov(t1, t2, eris)
    return Fme

def Woooo(t1, t2, eris):
    tau = make_tau(t2, t1, t1, eris)
    Wmnij = 0.25*einsum('ijef,mnef->mnij', tau, eris.oovv)
    Wmnij += cc_Woooo(t1, t2, eris)
    return Wmnij

def Wvvvv(t1, t2, eris):
    tau = make_tau(t2, t1, t1, eris)
    Wabef = cc_Wvvvv(t1, t2, eris)
    Wabef += einsum('mnab,mnef->abef', tau, .25*eris.oovv)
    return Wabef

def Wovvo(t1, t2, eris):
    Wmbej = -0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej += cc_Wovvo(t1, t2, eris)
    return Wmbej

def Wooov(t1, t2, eris):
    Wmnie = einsum('if,mnfe->mnie', t1, eris.oovv)
    Wmnie += eris.ooov
    return Wmnie

def Wvovv(t1, t2, eris):
    Wamef = einsum('na,nmef->amef', -t1, eris.oovv)
    Wamef -= eris.ovvv.transpose(1,0,2,3)
    return Wamef

def Wovoo(t1, t2, eris):
    tmp1 = einsum('mnie,jnbe->mbij', eris.ooov, t2)
    tmp2 = -einsum('ie,mbje->mbij', t1, eris.ovov)
    tmp = einsum('ie,mnef->mnif', t1, eris.oovv)
    tmp2 -= einsum('mnif,njbf->mbij', tmp, t2)

    FFov = Fov(t1, t2, eris)
    WWoooo = Woooo(t1, t2, eris)
    tau = make_tau(t2, t1, t1, eris)
    Wmbij = einsum('me,ijbe->mbij', -FFov, t2)
    Wmbij -= einsum('nb,mnij->mbij', t1, WWoooo)
    Wmbij += 0.5 * einsum('mbef,ijef->mbij', eris.ovvv, tau)
    Wmbij += tmp1 - tmp1.transpose(0,1,3,2)
    Wmbij += tmp2 - tmp2.transpose(0,1,3,2)
    Wmbij += eris.ooov.conj().transpose(2,3,0,1)
    return Wmbij

def Wvvvo(t1, t2, eris, Wvvvv_=None):
    tmp1 = einsum('mbef,miaf->abei', eris.ovvv, t2)
    tmp2 = einsum('ma,mbie->abei', t1, -eris.ovov)

    tmp = einsum('ma,mnef->anef', t1, eris.oovv)
    tmp2 -= einsum('nibf,anef->abei', t2, tmp)
    FFov = Fov(t1, t2, eris)
    tau = make_tau(t2, t1, t1, eris)
    Wabei  = 0.5 * einsum('mnie,mnab->abei', -eris.ooov, tau)
    Wabei -= einsum('me,miab->abei', FFov, t2)
    Wabei -= eris.ovvv.transpose(2,3,1,0).conj()
    Wabei -= tmp1 - tmp1.transpose(1,0,2,3)
    Wabei -= tmp2 - tmp2.transpose(1,0,2,3)
    if Wvvvv_ is None:
        Wvvvv_ = Wvvvv(t1, t2, eris)
    Wabei += einsum('abef,if->abei', Wvvvv_, t1)
    return Wabei

def get_t3p2_imds_slow(cc, t1, t2, eris=None, t3p2_ip_out=None, t3p2_ea_out=None):
    if eris is None:
        eris = cc.ao2mo()
    nocc, nvir = t1.shape

    fov = eris.fov

    oovv = eris.oovv
    ovvv = eris.ovvv
    ooov = eris.ooov
    oovv = eris.oovv
    vooo = ooov.conj().transpose(3,2,1,0)
    vvvo = ovvv.conj().transpose(3,2,1,0)

    ccsd_energy = cc.energy(t1, t2, eris)
    dtype = np.result_type(t1.dtype, t2.dtype)
    if np.issubdtype(dtype, np.dtype(complex).type):
        logger.error(cc, 't3p2 imds has not been strictly checked for use with complex integrals')

    tmp_t3 = einsum('bcdk,ijad->ijkabc', vvvo, t2)
    tmp_t3 -= einsum('cmkj,imab->ijkabc', vooo, t2)

    # P(ijk)
    tmp_t3 = (tmp_t3 + tmp_t3.transpose(1,2,0,3,4,5) +
                       tmp_t3.transpose(2,0,1,3,4,5))
    # P(abc)
    tmp_t3 = (tmp_t3 + tmp_t3.transpose(0,1,2,4,5,3) +
                       tmp_t3.transpose(0,1,2,5,3,4))

    eia = eris.eia
    eijab = eris.eijab
    eijkabc = eris.get_eijkabc()
    tmp_t3 /= eijkabc

    pt1 = 0.25 * einsum('mnef,imnaef->ia', oovv, tmp_t3)

    pt2 = einsum('ijmabe,me->ijab', tmp_t3, fov)
    tmp = einsum('ijmaef,mbfe->ijab', tmp_t3, ovvv)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)  # P(ab)
    tmp *= 0.5
    pt2 = pt2 + tmp

    tmp = einsum('inmabe,nmje->ijab', tmp_t3, ooov)
    tmp *= -0.5
    tmp = tmp - tmp.transpose(1, 0, 2, 3)  # P(ij)
    pt2 = pt2 + tmp

    pt1 /= eia
    pt2 /= eijab

    pt1 = pt1 + t1
    pt2 = pt2 + t2

    Wmcik = 0.5*einsum('ijkabc,mjab->mcik', tmp_t3, oovv)
    Wacek = -0.5*einsum('ijkabc,ijeb->acek', tmp_t3, oovv)

    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)
    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek
