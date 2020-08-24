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
from pyscf.lib import logger
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.cc.kccsd_t_rhf  import _get_epqr
from symtensor.ctf import zeros, einsum, frombatchfunc
from symtensor.ctf.backend import asarray

def get_t3p2_imds_slow(cc, t1, t2, eris=None):
    cpu0 = (time.clock(), time.time())
    if eris is None:
        eris = cc.ao2mo()

    kpts=  cc.kpts
    nocc, nvir = t1.shape
    nkpts = cc.nkpts
    dtype = np.result_type(t1, t2)

    fov = eris.fov
    mo_e_o = np.array([eris.mo_energy[ki][:nocc] for ki in range(nkpts)])
    mo_e_v = np.array([eris.mo_energy[ki][nocc:] for ki in range(nkpts)])

    # Get location of padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
    ccsd_energy = cc.energy(t1, t2, eris)

    Soovv = 2 * eris.ovov.transpose(0,2,1,3) - eris.ovov.transpose(0,2,3,1)
    ooov = eris.ooov.transpose(0,2,1,3).conj()
    ovvv = eris.ovvv.conj()

    def get_w(ki, kj, kk):
        ovvv_tmp  = ovvv[ki]
        t2_tmp = t2[kk,kj]
        w = einsum('iafb,kjcf->ijkabc', ovvv_tmp, t2_tmp)
        ooov_tmp = ooov[kj,ki]
        t2_tmp = t2[:,kk]
        w -= einsum('jima,mkbc->ijkabc', ooov_tmp, t2_tmp)
        return w

    def get_eijkabc(ki,kj,kk):
        eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                         [0,nocc,kj,mo_e_o,nonzero_opadding],
                         [0,nocc,kk,mo_e_o,nonzero_opadding])
        eijk = asarray(eijk)
        def _get_abc(ka,kb):
            kc = kpts_helper.get_kconserv3(cc._scf.cell, kpts, [ki,kj,kk,ka,kb])
            eabc = _get_epqr([0,nvir,ka,mo_e_v,nonzero_vpadding],
                             [0,nvir,kb,mo_e_v,nonzero_vpadding],
                             [0,nvir,kc,mo_e_v,nonzero_vpadding])
            ind = (ka*nkpts+kb)*eabc.size +np.arange(eabc.size)
            return ind, eabc.ravel()
        all_tasks = [[ka,kb] for ka,kb in itertools.product(range(nkpts), repeat=2)]
        shape = (nkpts,nkpts,nvir,nvir,nvir)
        eabc = frombatchfunc(_get_abc, shape, all_tasks).array
        eijkabc = eijk.reshape(1,1,nocc,nocc,nocc,1,1,1) -\
                  eabc.reshape(nkpts,nkpts,1,1,1,nvir,nvir,nvir)
        return eijkabc

    pt1 = zeros([nocc,nvir], sym=t1.sym, dtype=dtype)
    pt2 = zeros([nocc,nocc,nvir,nvir], sym=t2.sym, dtype=dtype)
    Woovo = zeros([nocc,nocc,nvir,nocc], sym=cc.gen_sym("++--"), dtype=dtype)
    Wovvv = zeros([nocc,nvir,nvir,nvir], sym=cc.gen_sym("++--"), dtype=dtype)

    for ki in range(nkpts):
        pt1tmp = zeros([nocc,nvir], dtype=dtype)
        pt2tmp_ja = zeros([nkpts,nkpts,nocc,nocc,nvir,nvir], dtype=dtype)
        Wovvvtmp = zeros([nkpts,nkpts,nocc,nvir,nvir,nvir], dtype=dtype)
        for kj in range(nkpts):
            pt2tmp_a = zeros([nkpts,nocc,nocc,nvir,nvir], dtype=dtype)
            Woovotmp = zeros([nkpts,nocc,nocc,nvir,nocc], dtype=dtype)
            for kk in range(nkpts):
                tmp_t3  = get_w(ki, kj, kk)
                tmp_t3 += get_w(ki, kk, kj).transpose(0, 2, 1, 3, 5, 4)
                tmp_t3 += get_w(kj, ki, kk).transpose(1, 0, 2, 4, 3, 5)
                tmp_t3 += get_w(kj, kk, ki).transpose(2, 0, 1, 5, 3, 4)
                tmp_t3 += get_w(kk, ki, kj).transpose(1, 2, 0, 4, 5, 3)
                tmp_t3 += get_w(kk, kj, ki).transpose(2, 1, 0, 5, 4, 3)
                d3 = get_eijkabc(ki,kj,kk)
                tmp_t3 /= d3

                St3 = tmp_t3 - tmp_t3.transpose(0,1,2,4,3,5)
                pt1tmp += einsum('Emnef,Eimnaef->ia', Soovv[kj,kk], St3[ki])

                ooov_tmp = eris.ooov[kj,:,kk]
                pt2tmp_ja -= 2*einsum('imnabe,mjne->ijab', tmp_t3, ooov_tmp)
                pt2tmp_ja += einsum('inmaeb,njme->ijab', tmp_t3, ooov_tmp)
                pt2tmp_ja += einsum('imneba,mjne->ijab', tmp_t3, ooov_tmp)

                tmp_t3t = tmp_t3.transpose(0,1,2,3,5,4)

                pt2tmp_a += einsum('Aijmaeb,me->Aijab', tmp_t3t[:,kk],fov[kk])
                pt2tmp_a -= einsum('Aijmaeb,me->Aijab', tmp_t3[:,kk],fov[kk])

                ovvv_tmp = eris.ovvv[kk]
                pt2tmp_a += 2*einsum('ijmaef,mfbe->ijab',tmp_t3, ovvv_tmp)
                pt2tmp_a -= einsum('ijmaef,mebf->ijab',tmp_t3, ovvv_tmp)
                pt2tmp_a -= einsum('ijmfea,mfbe->ijab',tmp_t3, ovvv_tmp)

                ovov_tmp  =eris.ovov[:,:,kk]
                Woovotmp += 2. * einsum('ijkabc,makc->jibm', tmp_t3, ovov_tmp)
                Woovotmp -= einsum('ijkacb,makc->jibm', tmp_t3, ovov_tmp)
                Woovotmp -= einsum('ijkcba,makc->jibm', tmp_t3, ovov_tmp)

                ovov_tmp = eris.ovov[kj,:,kk]
                Wovvvtmp -= 2. * einsum('ijkabc,jbke->ieac', tmp_t3, ovov_tmp)
                Wovvvtmp +=      einsum('ijkbac,jbke->ieac', tmp_t3, ovov_tmp)
                Wovvvtmp +=      einsum('ijkacb,jbke->ieac', tmp_t3, ovov_tmp)

            Woovo.array[kj,ki] = Woovotmp.array
            pt2.array[ki,kj] = pt2tmp_a.array

        Wovvv.array[ki] = Wovvvtmp.array
        pt1.array[ki] = pt1tmp.array
        pt2.array[ki] = pt2.array[ki] + pt2tmp_ja.array

    pt2 += pt2.transpose(1,0,3,2)
    pt1  = pt1/eris.eia + t1
    pt2  = pt2/eris.eijab + t2

    Wmcik = Woovo.transpose(3,2,1,0)
    Wacek = Wovvv.transpose(3,2,1,0)
    logger.timer(cc, 'EOM-CCSD(T) imds', *cpu0)
    delta_ccsd_energy = cc.energy(pt1, pt2, eris) - ccsd_energy
    logger.info(cc, 'CCSD energy T3[2] correction : %16.12e', delta_ccsd_energy)
    return delta_ccsd_energy, pt1, pt2, Wmcik, Wacek
