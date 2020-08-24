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
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.mp.kmp2 import padding_k_idx
from pyscf.pbc.lib import kpts_helper

from pyscf.ctfcc import kccsd_rhf
from symtensor.ctf import array, einsum, frombatchfunc
from symtensor.ctf.backend import asarray, dot

"""
See pyscf.pbc.cc.kccsd_t_rhf
"""

def kernel(mycc, eris, t1=None, t2=None, slice_size=None, free_vvvv=False):
    """
    See pyscf.pbc.cc.kccsd_t_rhf

    Args:
        slice_size:
            maximal memory(mb) allocated to t3 slicing, not the total maximal memory allowed.
            by default it's set to the same size as ovvv
        free_vvvv:
            a boolean for whether to free up vvvv integrals as it's not needed in (T)
    """
    assert isinstance(mycc, kccsd_rhf.KRCCSD)
    cpu1 = cpu0 = (time.clock(), time.time())

    log = logger.Logger(mycc.stdout, mycc.verbose)

    if eris is None:
        eris = mycc.ao2mo()

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2

    nkpts =  mycc.nkpts
    kpts = mycc.kpts
    nocc, nvir = t1.shape

    if slice_size is None:
        slice_size = nkpts**3*nocc*nvir**3
    else:
        slice_size = int(slice_size /4. * 6.25e4)

    fov = eris.fov
    t2t = t2.transpose(2,3,0,1) # make it easier for slicing
    vvov = eris.ovvv.conj().transpose(1,3,0,2) # <ac|ib> = (ia|bc)* Physicist's notation
    oovo = eris.ooov.conj().transpose(1,0,3,2) # (ij|ak) = (ji|ka)* Chemists's notation
    vvoo = eris.ovov.transpose(1,3,0,2).conj() # <ab|ij> = (ai|bj) = (ia|jb)*, Physicist's notation

    def get_w(ka,kb,kc, orbslice=None):
        '''Wijkabc intermediate as described in Scuseria paper before Pijkabc acts'''
        if orbslice is None:
            a0, a1, b0, b1, c0, c1 = 0, nvir, 0, nvir, 0, nvir
        else:
            a0, a1, b0, b1, c0, c1 = orbslice

        ttmp= t2[:,:,kc,:,:,c0:c1]
        wtmp = vvov[ka,kb,:,a0:a1,b0:b1]
        w = einsum('kjcf,abif->abcijk', ttmp, wtmp)
        ttmp = t2t[kb,kc,:,b0:b1,c0:c1]
        wtmp = oovo[:,:,ka,:,:,a0:a1]
        w-= einsum('bcmk,mjai->abcijk', ttmp, wtmp)
        return w

    def get_v(ka,kb,kc,orbslice=None):
        '''Vijkabc intermediate as described in Scuseria paper'''
        if orbslice is None:
            a0, a1, b0, b1, c0, c1 = 0, nvir, 0, nvir, 0, nvir
        else:
            a0, a1, b0, b1, c0, c1 = orbslice
        sym = mycc.gen_sym("++", mycc.kpts[ka]+mycc.kpts[kb])
        irrep_map = mycc.symlib.get_irrep_map(sym)
        v = einsum('kc,Iabij->Iabcijk', t1[kc,:,c0:c1], vvoo[ka,kb,:,a0:a1,b0:b1])
        v+= einsum('kc,Iabij->Iabcijk', fov[kc,:,c0:c1], t2t[ka,kb,:,a0:a1,b0:b1])
        v = einsum('Iabcijk,IJ->IJabcijk', v, irrep_map)
        sym = mycc.gen_sym("000+++", mycc.kpts[ka]+mycc.kpts[kb]+mycc.kpts[kc])
        return array(v, sym)

    mo_e_o = [eris.mo_energy[ki][:nocc] for ki in range(nkpts)]
    mo_e_v = [eris.mo_energy[ki][nocc:] for ki in range(nkpts)]


    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    def get_d3(ka,kb,kc,orbslice=None):
        if orbslice is None:
            a0, a1, b0, b1, c0, c1 = 0, nvir, 0, nvir, 0, nvir
        else:
            a0, a1, b0, b1, c0, c1 = orbslice

        eabc = _get_epqr([a0,a1,ka,mo_e_v,nonzero_vpadding],
                         [b0,b1,kb,mo_e_v,nonzero_vpadding],
                         [c0,c1,kc,mo_e_v,nonzero_vpadding],
                         fac=[-1.,-1.,-1.])

        def _get_eabcijk(ki,kj):
            kk = kpts_helper.get_kconserv3(mycc._scf.cell, kpts, [ka, kb, kc, ki, kj])
            eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                             [0,nocc,kj,mo_e_o,nonzero_opadding],
                             [0,nocc,kk,mo_e_o,nonzero_opadding])

            eabcijk  = (eabc[:,:,:,None,None,None] + eijk[None,None,None,:,:,:])
            ind = (ki*nkpts+kj)*eabcijk.size+np.arange(eabcijk.size)
            return ind, eabcijk.ravel()

        all_tasks  = [[ki,kj] for ki,kj in itertools.product(range(nkpts), repeat=2)]
        eout = frombatchfunc(_get_eabcijk, \
                (nkpts,nkpts,a1-a0,b1-b0,c1-c0,nocc,nocc,nocc), all_tasks).array
        return eout

    energy_t = 0.
    blkmin = 4
    vir_blksize = min(nvir, max(blkmin, int((slice_size/nkpts**2)**(1./3)/nocc)))
    log.info("nvir = %i, virtual slice size: %i", nvir, vir_blksize)
    tasks = []
    for a0, a1 in lib.prange(0, nvir, vir_blksize):
        for b0, b1 in lib.prange(0, nvir, vir_blksize):
            for c0, c1 in lib.prange(0, nvir, vir_blksize):
                tasks.append((a0,a1,b0,b1,c0,c1))

    for ka, kb, kc in itertools.product(range(nkpts), repeat=3):
        '''only iterate over ka>=kb>=kc'''
        if not (ka >= kb and kb >= kc):
            continue
        symm_kpt = max((ka!=kb)*3+(kb!=kc)*3, 1)
        for orbslice in tasks:
            a0, a1, b0, b1, c0, c1 = orbslice
            '''when the virtual slicing allows, only iterate over ka,a>=kb,b>=kc,c'''
            if ka==kb and kb==kc:
                if not (a0>=b0 and b0>=c0): continue
                symm_abc = max((a0!=b0)*3+(b0!=c0)*3, 1)
            elif ka==kb:
                if not (a0>=b0): continue
                symm_abc = max((a0!=b0)*2, 1)
            elif kb==kc:
                if not (b0>=c0): continue
                symm_abc = max((b0!=c0)*2, 1)
            else:
                symm_abc = 1

            symm = symm_kpt * symm_abc
            permuted_w = get_w(ka,kb,kc, orbslice)
            permuted_w += get_w(kb,kc,ka,[b0,b1,c0,c1,a0,a1]).transpose(2,0,1,5,3,4)
            permuted_w += get_w(kc,ka,kb,[c0,c1,a0,a1,b0,b1]).transpose(1,2,0,4,5,3)
            permuted_w += get_w(ka,kc,kb,[a0,a1,c0,c1,b0,b1]).transpose(0,2,1,3,5,4)
            permuted_w += get_w(kc,kb,ka,[c0,c1,b0,b1,a0,a1]).transpose(2,1,0,5,4,3)
            permuted_w += get_w(kb,ka,kc,[b0,b1,a0,a1,c0,c1]).transpose(1,0,2,4,3,5)

            rw = 4*permuted_w + permuted_w.transpose(0,1,2,5,3,4) + \
                   permuted_w.transpose(0,1,2,4,5,3) - 2*permuted_w.transpose(0,1,2,3,5,4) - \
                 2*permuted_w.transpose(0,1,2,5,4,3) - 2*permuted_w.transpose(0,1,2,4,3,5)

            permuted_w += get_v(ka,kb,kc, orbslice) *.5
            permuted_w += get_v(kb,kc,ka,[b0,b1,c0,c1,a0,a1]).transpose(2,0,1,5,3,4) *.5
            permuted_w += get_v(kc,ka,kb,[c0,c1,a0,a1,b0,b1]).transpose(1,2,0,4,5,3) *.5
            permuted_w += get_v(ka,kc,kb,[a0,a1,c0,c1,b0,b1]).transpose(0,2,1,3,5,4) *.5
            permuted_w += get_v(kc,kb,ka,[c0,c1,b0,b1,a0,a1]).transpose(2,1,0,5,4,3) *.5
            permuted_w += get_v(kb,ka,kc,[b0,b1,a0,a1,c0,c1]).transpose(1,0,2,4,3,5) *.5
            eabcijk = get_d3(ka,kb,kc,orbslice)

            energy_t += symm *einsum('abcijk,abcijk->', permuted_w, rw.conj()/eabcijk) / 3 / nkpts

    if abs(energy_t.imag) > 1e-4:
        log.warn('Non-zero imaginary part of CCSD(T) energy was found %s', energy_t.imag)
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction per cell = %.15g', energy_t.real)
    log.note('CCSD(T) correction per cell (imag) = %.15g', energy_t.imag)
    return energy_t.real

def ipccsd_star_contract(mycc, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift):
    assert(mycc.ip_partition == None)
    if not mycc.imds.made_ip_imds:
        mycc.imds.make_ip()

    eris = mycc.imds.eris
    t1, t2 = mycc.t1, mycc.t2
    nkpts=  mycc.nkpts
    kpts= mycc.kpts
    nocc, nvir = t1.shape

    mo_e_o = [eris.mo_energy[ki][:nocc] for ki in range(nkpts)]
    mo_e_v = [eris.mo_energy[ki][nocc:] for ki in range(nkpts)]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    abij = eris.ovov.transpose(1,3,0,2)
    abie = eris.ovvv.transpose(1,3,0,2)
    bjmk = eris.ooov.transpose(3,2,1,0)
    vovv = eris.ovvv.conj().transpose(1,0,3,2)
    vooo = bjmk.conj()
    t2t = t2.transpose(2,3,0,1)


    def get_eabijk(ka,kb):
        eab = _get_epq([0,nvir,ka,mo_e_v,nonzero_vpadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[-1.,-1.])
        eab = asarray(eab)

        def _get_ijk(ki,kj):
            kk = kpts_helper.get_kconserv3(mycc._scf.cell, kpts, [ka, kb, kshift, ki, kj])
            eijk = _get_epqr([0,nocc,ki,mo_e_o,nonzero_opadding],
                             [0,nocc,kj,mo_e_o,nonzero_opadding],
                             [0,nocc,kk,mo_e_o,nonzero_opadding])
            off = ki *nkpts+kj
            ind = off *eijk.size + np.arange(eijk.size)
            return ind, eijk.ravel()
        all_tasks = [[ki,kj] for ki,kj in itertools.product(range(nkpts), repeat=2)]
        shape = (nkpts,nkpts,nocc,nocc,nocc)
        eijk = frombatchfunc(_get_ijk, shape, all_tasks).array
        eabijk = eab.reshape(1,1,nvir,nvir,1,1,1) + eijk.reshape(nkpts,nkpts,1,1,nocc,nocc,nocc)
        return eabijk

    def get_labijk(ka,kb,l1,l2t):
        vvoo_tmp = abij[ka,kb]
        vvov_tmp = abie[ka,kb]
        vooo_tmp = bjmk[kb]
        l2t_tmp  = l2t[ka]
        labijk = 0.5*einsum('abij,k->abijk', vvoo_tmp, l1)
        labijk += einsum('abie,ejk->abijk', vvov_tmp, l2t)
        labijk += -einsum('bjmk,aim->abijk', vooo_tmp, l2t_tmp)
        labijk += -einsum('bjmi,amk->abijk', vooo_tmp, l2t_tmp)
        return labijk

    def get_rabijk(ka,kb,r1,r2t,iooo):
        oovv_tmp = eris.oovv[:,:,kb]
        ovvo_tmp = eris.ovvo[:,:,kb]
        vovv_tmp = vovv[ka,:,kb]
        vooo_tmp = vooo[kb]

        t2a_tmp = t2t[ka]
        t2ab_tmp = t2t[ka,kb]
        r2t_tmp = r2t[ka]

        rabijk = -einsum('mkbe,m,aeij->abijk', oovv_tmp, r1, t2a_tmp)
        rabijk -= einsum('mebj,m,aeik->abijk', ovvo_tmp, r1, t2a_tmp)
        rabijk += einsum('mjk,abim->abijk', iooo, t2ab_tmp)
        rabijk +=  einsum('aibe,ekj->abijk', vovv_tmp, r2t)
        rabijk += -einsum('bjmk,ami->abijk', vooo_tmp, r2t_tmp)
        rabijk += -einsum('bjmi,akm->abijk', vooo_tmp, r2t_tmp)
        return rabijk

    e = []
    nroots = len(ipccsd_evals)
    for k in range(nroots):
        l1, l2 = mycc.vector_to_amplitudes_ip(lipccsd_evecs[k], kshift=kshift)
        r1, r2 = mycc.vector_to_amplitudes_ip(ipccsd_evecs[k], kshift=kshift)
        ldotr = dot(l1.ravel().conj(),r1.ravel()) + dot(l2.ravel(),r2.ravel())
        logger.info(mycc, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real(), ldotr.imag())
        if abs(ldotr) < 1e-7:
            logger.warn(mycc, 'Small %s left-right amplitude overlap for %i th root. Results '
                             'may be inaccurate.', abs(ldotr), k)

        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(l2 + 2.*l2.transpose(1,0,2))

        r2t = r2.transpose(2,0,1)
        l2t = l2.transpose(2,0,1)
        ooo = einsum('mjnk,n->mjk', eris.oooo, r1)
        deltaE = 0.
        for ka in range(nkpts):
            for kb in range(ka, nkpts):
                fac = 2 - (ka==kb)
                labijk = get_labijk(ka,kb,l1,l2t)
                labijk += get_labijk(kb,ka,l1,l2t).transpose(1,0,3,2,4)
                labijk = 4.*labijk \
                        - 2.*labijk.transpose(0,1,3,2,4) \
                        - 2.*labijk.transpose(0,1,4,3,2) \
                        - 2.*labijk.transpose(0,1,2,4,3) \
                        + 1.*labijk.transpose(0,1,3,4,2) \
                        + 1.*labijk.transpose(0,1,4,2,3)
                rabijk = get_rabijk(ka,kb,r1,r2t, ooo)
                rabijk += get_rabijk(kb,ka,r1,r2t, ooo).transpose(1,0,3,2,4)
                d3 = get_eabijk(ka,kb) + ipccsd_evals[k]
                deltaE += 0.5*einsum('abijk,abijk',labijk,rabijk/d3) * fac

        deltaE = deltaE.real
        logger.note(mycc, "Exc. energy, delta energy = %16.12f, %16.12f",
                    ipccsd_evals[k]+deltaE, deltaE)
        e.append(ipccsd_evals[k]+deltaE)

    return e

def eaccsd_star_contract(mycc, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift):
    assert(mycc.ea_partition == None)
    if not mycc.imds.made_ea_imds:
        mycc.imds.make_ea()

    eris = mycc.imds.eris
    t1, t2 = mycc.t1, mycc.t2
    nkpts=  mycc.nkpts
    kpts= mycc.kpts
    nocc, nvir = t1.shape

    mo_e_o = [eris.mo_energy[ki][:nocc] for ki in range(nkpts)]
    mo_e_v = [eris.mo_energy[ki][nocc:] for ki in range(nkpts)]
    nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")

    vovv = eris.ovvv.conj().transpose(1,0,3,2)
    jbem = eris.ovvo.conj()
    jmia = eris.ooov.conj()

    def get_eijabc(ki,kj):
        all_tasks = [[ka,kb] for ka,kb in itertools.product(range(nkpts),repeat=2)]
        eij = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nocc,kj,mo_e_o,nonzero_opadding])
        eij = asarray(eij)
        def _get_eabc(ka,kb):
            kc = kpts_helper.get_kconserv3(mycc._scf.cell, kpts, [ki,kj,kshift,ka,kb])
            eabc = _get_epqr([0,nvir,ka,mo_e_v,nonzero_vpadding],
                             [0,nvir,kb,mo_e_v,nonzero_vpadding],
                             [0,nvir,kc,mo_e_v,nonzero_vpadding],
                             fac=[-1., -1., -1.])
            off = ka*nkpts+kb
            ind = off*eabc.size+np.arange(eabc.size)
            return ind, eabc.ravel()
        shape = (nkpts,nkpts,nvir,nvir,nvir)
        eabc = frombatchfunc(_get_eabc, shape, all_tasks).array
        eijabc = eij.reshape(1,1,nocc,nocc,1,1,1) + \
                 eabc.reshape(nkpts,nkpts,1,1,nvir,nvir,nvir)
        return eijabc

    def get_lijabc(ki,kj,l1,l2):
        ooov_tmp = eris.ooov[kj,:,ki]
        ovov_tmp = eris.ovov[ki,:,kj]
        lijabc = einsum('jmia,mbc->ijabc', ooov_tmp, l2)
        lijabc += -0.5*einsum('c,iajb->ijabc', l1, ovov_tmp)
        l2_tmp = l2[kj]
        ovvv_tmp = eris.ovvv[ki]
        lijabc -= einsum('iaeb,jec->ijabc', ovvv_tmp, l2_tmp)
        l2_tmp = l2[ki]
        ovvv_tmp = eris.ovvv[kj]
        lijabc -= einsum('jbec,iae->ijabc', ovvv_tmp, l2_tmp)
        return lijabc

    def get_rijabc(ki,kj,r1,r2,vvv):
        t2_tmp = t2[ki,kj]
        rijabc = -einsum('bec,ijae->ijabc', vvv, t2_tmp)
        oovv_tmp = eris.oovv[:,kj]
        t2_tmp = t2[ki]
        rijabc += einsum('mjce,e,imab->ijabc', oovv_tmp, r1, t2_tmp)
        ovvo_tmp = jbem[kj]
        rijabc += einsum('jbem,e,imac->ijabc', ovvo_tmp, r1, t2_tmp)
        ooov_tmp = jmia[kj,:,ki]
        rijabc += einsum('jmia,mbc->ijabc', ooov_tmp, r2)
        vovv_tmp = vovv[:,kj]
        r2_tmp = r2[ki]
        rijabc += -einsum('bjce,iae->ijabc', vovv_tmp, r2_tmp)
        vovv_tmp = vovv[:,ki]
        r2_tmp = r2[kj]
        rijabc += -einsum('aibe,jec->ijabc', vovv_tmp, r2_tmp)
        return rijabc

    e = []
    nroots = len(eaccsd_evals)
    for k in range(nroots):
        l1,l2 = mycc.vector_to_amplitudes_ea(leaccsd_evecs[k], kshift=kshift)
        r1,r2 = mycc.vector_to_amplitudes_ea(eaccsd_evecs[k], kshift=kshift)
        ldotr = dot(l1.ravel(),r1.ravel()) + dot(l2.ravel(),r2.ravel())
        logger.info(mycc, 'Left-right amplitude overlap : %14.8e + 1j %14.8e',
                    ldotr.real(), ldotr.imag())
        if abs(ldotr) < 1e-7:
            logger.warn(mycc, 'Small %s left-right amplitude overlap for %i th root. Results '
                             'may be inaccurate.', abs(ldotr), k)
        l1 /= ldotr
        l2 /= ldotr
        l2 = 1./3*(1.*l2 + 2.*l2.transpose(0,2,1))
        r2 = r2.transpose(0,2,1)

        vvv = einsum('becf,f->bec', eris.vvvv, r1)
        deltaE = 0.0 + 0.0j
        for ki in range(nkpts):
            for kj in range(ki,nkpts):
                fac = 2. - (ki==kj)
                lijabc = get_lijabc(ki,kj,l1,l2)
                lijabc+= get_lijabc(kj,ki,l1,l2).transpose(1,0,3,2,4)
                lijabc =  4.*lijabc \
                         - 2.*lijabc.transpose(0,1,3,2,4) \
                         - 2.*lijabc.transpose(0,1,4,3,2) \
                         - 2.*lijabc.transpose(0,1,2,4,3) \
                         + 1.*lijabc.transpose(0,1,3,4,2) \
                         + 1.*lijabc.transpose(0,1,4,2,3)
                rijabc = get_rijabc(ki,kj,r1,r2,vvv)
                rijabc+= get_rijabc(kj,ki,r1,r2,vvv).transpose(1,0,3,2,4)
                d3 = get_eijabc(ki,kj)
                d3 = 1./(d3+eaccsd_evals[k])
                deltaE += 0.5*einsum('ijabc,ijabc',lijabc,rijabc*d3) * fac

        deltaE = deltaE.real
        logger.note(mycc, "Exc. energy, delta energy = %16.12f, %16.12f",
                    eaccsd_evals[k]+deltaE, deltaE)
        e.append(eaccsd_evals[k]+deltaE)
    return e

if __name__ == '__main__':
    from pyscf.pbc import gto, scf, cc
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
    kpts=cell.make_kpts([1, 1, 3])
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    kmf.kernel()

    mycc = kccsd_rhf.KRCCSD(kmf)
    eris = mycc.ao2mo()
    mycc.kernel()
    et1 = kernel(mycc, eris)
    print(et1 - -5.1848049827171756e-06)
