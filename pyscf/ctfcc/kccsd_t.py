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

import numpy as np
import time
import itertools
from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc.cc.kccsd_t_rhf  import _get_epqr
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.pbc.lib import kpts_helper

from pyscf.ctfcc import kccsd
from symtensor.ctf import array, einsum, frombatchfunc
from symtensor.ctf.backend import asarray


def kernel(mycc, eris, t1=None, t2=None, slice_size=None, free_vvvv=False):
    """
    See pyscf.pbc.cc.kccsd_t
    slice_size:
        the largest size of temporary t3t one wants to keep in memory(MB), if not set, it will match the size of eris.ovvv
    free_vvvv:
        whether to free up the vvvv integrals in eris object
    """
    assert isinstance(mycc, kccsd.KGCCSD)
    cpu1 = cpu0 = (time.clock(), time.time())

    log = logger.Logger(mycc.stdout, mycc.verbose)
    if eris is None:
        eris = mycc.ao2mo()
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nkpts =  mycc.nkpts
    cell = mycc._scf.cell
    kpts = mycc.kpts
    nocc, nvir = t1.shape

    if slice_size is None:
        slice_size = nkpts**3*nocc*nvir**3
    else:
        slice_size = int(slice_size /4. * 6.25e4)

    blkmin = 4
    vir_blksize = min(nvir, max(blkmin, int((slice_size/nkpts**2)**(1./3)/nocc)))
    log.info("nvir = %i, virtual slice size: %i", nvir, vir_blksize)
    tasks = []
    for a0, a1 in lib.prange(0, nvir, vir_blksize):
        for b0, b1 in lib.prange(0, nvir, vir_blksize):
            for c0, c1 in lib.prange(0, nvir, vir_blksize):
                if b0>=a0 and c0>=b0:
                    tasks.append((a0,a1,b0,b1,c0,c1))

    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")
    mo_e_o = [eris.mo_energy[ki][:nocc] for ki in range(nkpts)]
    mo_e_v = [eris.mo_energy[ki][nocc:] for ki in range(nkpts)]

    iecb = eris.ovvv.conj()
    jkma = eris.ooov.conj()
    jkbc = eris.oovv.conj()
    fvo = eris.fov.transpose(1,0).conj()

    def compute_t3c(ki,kj,kk,a0,a1,b0,b1,c0,c1):
        ttmp = t2[kj,kk,:,:,:,a0:a1]
        wtmp = iecb[ki,:,:,:,:,c0:c1,b0:b1]
        t3c = einsum('jkae,iecb->ijkabc', ttmp, wtmp)
        ttmp = t2[ki,:,:,:,:,b0:b1,c0:c1]
        wtmp = jkma[kj,kk,:,:,:,:,a0:a1]
        t3c -= einsum('imbc,jkma->ijkabc', ttmp, wtmp)
        return t3c

    def compute_t3d(ki,kj,kk,a0,a1,b0,b1,c0,c1):
        kia = mycc.symlib.get_irrep_map(mycc.gen_sym("+", kpts[ki]))
        ttmp = einsum('Iia,I->Iia', t1[:,:,a0:a1], kia)
        ftmp = einsum('Iai,I->Iai', fvo[:,a0:a1], kia)
        t3d = einsum('Aia,Bjkbc->ABijkabc', ttmp, jkbc[kj,kk,:,:,:,b0:b1,c0:c1])
        t3d += einsum('Aai,Bjkbc->ABijkabc', ftmp, t2[kj,kk,:,:,:,b0:b1,c0:c1])
        sym = mycc.gen_sym("000+++", kpts[ki]+kpts[kj]+kpts[kk])
        t3d = array(t3d, sym)
        t3d.symlib = mycc.symlib
        return t3d

    def permute_abc(func,ki,kj,kk,a0,a1,b0,b1,c0,c1):
        out = func(ki,kj,kk,a0,a1,b0,b1,c0,c1) - \
              func(ki,kj,kk,b0,b1,a0,a1,c0,c1).transpose(0,1,2,4,3,5) - \
              func(ki,kj,kk,c0,c1,b0,b1,a0,a1).transpose(0,1,2,5,4,3)
        return out

    def permute_ijk(func,ki,kj,kk,a0,a1,b0,b1,c0,c1):
        out = func(ki,kj,kk,a0,a1,b0,b1,c0,c1)
        out -= func(kj,ki,kk,a0,a1,b0,b1,c0,c1).transpose(1,0,2,3,4,5)
        out -= func(kk,kj,ki,a0,a1,b0,b1,c0,c1).transpose(2,1,0,3,4,5)
        return out

    def get_eijkabc(ki,kj,kk,a0,a1,b0,b1,c0,c1):
        sym = mycc.gen_sym("+++", kpts[ki]+kpts[kj]+kpts[kk])
        eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                         [0,nocc,kj,mo_e_o,nonzero_opadding],
                         [0,nocc,kk,mo_e_o,nonzero_opadding])
        eijk = asarray(eijk)
        def _get_abc(ka,kb):
            kc = kpts_helper.get_kconserv3(cell, kpts, [ki, kj, kk, ka, kb])
            eabc = _get_epqr([a0,a1,ka,mo_e_v,nonzero_vpadding],
                             [b0,b1,kb,mo_e_v,nonzero_vpadding],
                             [c0,c1,kc,mo_e_v,nonzero_vpadding])
            ind = (ka*nkpts+kb)*eabc.size +np.arange(eabc.size)
            return ind, eabc.ravel()
        all_tasks = [[ka,kb] for ka,kb in itertools.product(range(nkpts), repeat=2)]
        shape = (a1-a0,b1-b0,c1-c0)
        eabc = frombatchfunc(_get_abc, shape, all_tasks, sym=sym).array
        eijkabc = eijk.reshape(1,1,nocc,nocc,nocc,1,1,1) -\
                  eabc.reshape(nkpts,nkpts,1,1,1,a1-a0,b1-b0,c1-c0)
        sym = mycc.gen_sym("000+++", kpts[ki]+kpts[kj]+kpts[kk])
        return array(eijkabc, sym)

    def compute_permuted_t3c(ki,kj,kk,a0,a1,b0,b1,c0,c1):
        func = lambda *args: permute_abc(compute_t3c,*args)
        gen_func = lambda *args: permute_ijk(func, *args)
        t3c = gen_func(ki,kj,kk,a0,a1,b0,b1,c0,c1)
        return t3c

    def compute_permuted_t3d(ki,kj,kk,a0,a1,b0,b1,c0,c1):
        func = lambda *args: permute_abc(compute_t3d,*args)
        gen_func = lambda *args: permute_ijk(func, *args)
        t3d = gen_func(ki,kj,kk,a0,a1,b0,b1,c0,c1)
        return t3d

    energy_t = 0.0
    for ki in range(nkpts):
        for kj in range(ki + 1):
            for kk in range(kj + 1):
                if ki == kj and kj == kk:
                    symm_ijk_kpt = 1.  # only one degeneracy
                elif ki == kj or kj == kk:
                    symm_ijk_kpt = 3.  # 3 degeneracies when only one k-point is unique
                else:
                    symm_ijk_kpt = 6.  # 3! combinations of arranging 3 distinct k-points
                for (a0, a1, b0, b1, c0, c1) in tasks:
                    if a0==c0:
                        sym_abc = 1.
                    elif a0==b0 or b0==c0:
                        sym_abc = 3.
                    else:
                        sym_abc = 6.
                    d3 = get_eijkabc(ki,kj,kk,a0,a1,b0,b1,c0,c1)
                    t3c = compute_permuted_t3c(ki,kj,kk,a0,a1,b0,b1,c0,c1)
                    t3d = compute_permuted_t3d(ki,kj,kk,a0,a1,b0,b1,c0,c1)
                    energy_t+= einsum('ijkabc,ijkabc', (t3c+t3d).conj()/d3, t3c) / 36 * symm_ijk_kpt * sym_abc

    energy_t /= mycc.nkpts
    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s',
                 energy_t.imag)
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
    log.note('CCSD(T) correction per cell (imag) = %.15g', energy_t.imag)

    return energy_t.real


if __name__ == '__main__':
    from pyscf.pbc import gto, scf
    from pyscf.ctfcc import kccsd
    cell = gto.Cell()
    cell.atom = '''
    He 0.000000000000   0.000000000000   0.000000000000
    He 1.685068664391   1.685068664391   1.685068664391
    '''
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.unit = 'B'
    cell.verbose = 4
    cell.build()

    # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
    kmf.kernel()
    kmf = kmf.to_ghf(kmf)
    mycc = kccsd.KGCCSD(kmf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    et1 = kernel(mycc, eris)
    print(et1 - -5.1848049827171756e-06)
