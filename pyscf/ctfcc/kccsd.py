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
KGCCSD with CTF as backend, all integrals in memory
'''

import numpy as np
import time
from functools import reduce
import itertools
from pyscf import lib

from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import gccsd_slow
from pyscf.pbc.mp.kmp2 import (padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
import pyscf.pbc.tools.pbc as tools
from pyscf.pbc.cc import kccsd, eom_kccsd_ghf
from pyscf.ctfcc import gccsd, kccsd_rhf, ctf_helper
from pyscf.ctfcc.integrals import ao2mo
from symtensor.ctf import einsum, array, frombatchfunc, zeros
from symtensor.ctf.backend import hstack, asarray, eye, argsort
from symtensor.symlib import SYMLIB

SLICE_SIZE = getattr(__config__, 'ctfcc_kccsd_slice_size', 4000)

rank = ctf_helper.rank

def energy(mycc, t1, t2, eris):
    return gccsd_slow.energy(mycc, t1, t2, eris, fac=1./mycc.nkpts)

def amplitudes_to_vector_ip(r1, r2):
    nkpts = r2.array.shape[0]
    nocc, nvir = r2.shape[1:]
    r2v = r2.array.transpose(0,2,1,3,4).reshape(nkpts*nocc,nkpts*nocc,nvir)
    r2v = ctf_helper.pack_ip_r2(r2v)
    return hstack((r1.ravel(), r2v))

def amplitudes_to_vector_ea(r1, r2):
    nkpts = r2.array.shape[0]
    nocc, nvir = r2.shape[:2]
    r2v = r2.transpose(1,2,0).array.transpose(0,2,1,3,4).reshape(nkpts*nvir,nkpts*nvir,nocc)
    r2v = ctf_helper.pack_ea_r2(r2v.transpose(2,0,1))
    return hstack((r1.ravel(), r2v))

class KCCSD(kccsd_rhf.KRCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, slice_size=SLICE_SIZE):
        ctf_helper.synchronize(mf, ["mo_coeff", "mo_occ", "mo_energy"])
        kccsd.KCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.ip_partition = self.ea_partition = None
        self.slice_size = SLICE_SIZE
        self.max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KCCSD_max_space', 20)
        self.symlib = SYMLIB('ctf')
        self.__imds__  = None
        self._keys = self._keys.union(['max_space', 'ip_partition', '__imds__'\
                                       'ea_partition', 'symlib', 'slice_size'])
        self.make_symlib()

    def init_amps(self, eris=None):
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        sym1 = self._sym[0]
        t1 = zeros([nocc,nvir], sym=sym1)
        t2 = eris.oovv.conj() / eris.eijab
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris.oovv).real / self.nkpts
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    dump_flags = kccsd.KGCCSD.dump_flags
    energy = energy
    update_amps = gccsd.GCCSD.update_amps
    ccsd = gccsd.GCCSD.ccsd
    kernel = gccsd.GCCSD.kernel

    ipccsd_matvec = gccsd.GCCSD.ipccsd_matvec
    eaccsd_matvec = gccsd.GCCSD.eaccsd_matvec
    lipccsd_matvec = gccsd.GCCSD.lipccsd_matvec
    leaccsd_matvec = gccsd.GCCSD.leaccsd_matvec
    solve_lambda = gccsd.GCCSD.solve_lambda

    nip = eom_kccsd_ghf.EOMIP.vector_size
    nea = eom_kccsd_ghf.EOMEA.vector_size

    def ao2mo(self, mo_coeff=None):
        return _PhysicistsERIs(self, mo_coeff)

    @property
    def imds(self):
        if self.__imds__ is None:
            self.__imds__ = gccsd_slow._IMDS(self)
        return self.__imds__

    def amplitudes_to_vector_ip(self, r1, r2, **kwargs):
        return amplitudes_to_vector_ip(r1, r2)

    def amplitudes_to_vector_ea(self, r1, r2, **kwargs):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_to_amplitudes_ip(self, vector, kshift=0):
        kpti = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpti)
        sym2 = self.gen_sym('++-', kpti)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        r1 = array(vector[:nocc], sym1)
        r2 = ctf_helper.unpack_ip_r2(vector[nocc:], nkpts*nocc+nvir, nkpts*nocc).reshape(nkpts,nocc,nkpts,nocc,nvir).transpose(0,2,1,3,4)
        r2 = array(r2, sym2)
        r1.symlib = r2.symlib = self.symlib
        return r1, r2

    def vector_to_amplitudes_ea(self, vector, kshift=0):
        kpta = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpta)
        sym2 = self.gen_sym('++-', kpta)
        nocc = self.nocc
        nvir = self.nmo - nocc
        nkpts = self.nkpts
        r1 = array(vector[:nvir], sym1)
        r2 = ctf_helper.unpack_ea_r2(vector[nvir:], nkpts*nvir+nocc, nocc).reshape(nocc,nkpts,nvir,nkpts,nvir).transpose(1,3,2,4,0)
        r2 = array(r2, sym2).transpose(2,0,1)
        r1.symlib = r2.symlib = self.symlib
        return r1, r2

    def ipccsd_diag(self, kshift):
        if not self.imds.made_ip_imds:
            self.imds.make_ip()
        imds = self.imds
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('++-', self.kpts[kshift])
        nkpts = self.nkpts
        nocc,nvir = imds.t1.shape
        t2 = imds.t2
        Hr1 = -imds.Foo.diagonal()[kshift]
        Hr1 = array(Hr1, sym1)
        IJA = self.symlib.get_irrep_map(sym2)
        if self.ip_partition == 'mp':
            foo = imds.eris.foo.diagonal()
            fvv = imds.eris.fvv.diagonal()
            Hr2 = (-foo.reshape(nkpts,1,nocc,1,1) - foo.reshape(1,nkpts,1,nocc,1) +\
                    einsum('Aa,IJA->IJa', fvv, IJA).reshape(nkpts,nkpts,1,1,nvir))
        else:
            foo = imds.Foo.diagonal()
            fvv = imds.Fvv.diagonal()
            Hr2 = (-foo.reshape(nkpts,1,nocc,1,1) - foo.reshape(1,nkpts,1,nocc,1) +\
                    einsum('Aa,IJA->IJa', fvv, IJA).reshape(nkpts,nkpts,1,1,nvir))
            Hr2 += (einsum('IJIijij->IJij', imds.Woooo).reshape(nkpts,nkpts,nocc,nocc,-1) +\
                    einsum('IAAiaai,IJA->IJia', imds.Wovvo, IJA).reshape(nkpts,nkpts,nocc,1,nvir) +\
                    einsum('JAAjaaj,IJA->IJja', imds.Wovvo, IJA).reshape(nkpts,nkpts,1,nocc,nvir) +\
                    einsum('IJijea,JIjiea->IJija', imds.Woovv[:,:,kshift], imds.t2[:,:,kshift]))
        Hr2 = array(Hr2, sym2)
        return self.amplitudes_to_vector_ip(Hr1, Hr2)

    def eaccsd_diag(self, kshift):
        if not self.imds.made_ea_imds:
            self.imds.make_ea()
        imds = self.imds
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('-++', self.kpts[kshift])
        nkpts = self.nkpts
        nocc,nvir = imds.t1.shape
        t2 = imds.t2
        Hr1 = array(imds.Fvv.diagonal()[kshift], sym1)
        IAB = self.symlib.get_irrep_map(sym2)
        if self.ea_partition == 'mp': # This case is untested
            foo = imds.eris.foo.diagonal()
            fvv = imds.eris.fvv.diagonal()
            Hr2 = (-foo.reshape(nkpts,1,nocc,1,1) + fvv.reshape(1,nkpts,1,nvir,1) +\
                    einsum('Bb,IAB->IAb', fvv, IAB).reshape(nkpts,nkpts,1,1,nvir))
        else:
            foo = imds.Foo.diagonal()
            fvv = imds.Fvv.diagonal()
            Hr2 = (-foo.reshape(nkpts,1,nocc,1,1) + fvv.reshape(1,nkpts,1,nvir,1) +\
                    einsum('Bb,IAB->IAb', fvv, IAB).reshape(nkpts,nkpts,1,1,nvir))
            Hr2 += (einsum('JBBjbbj,JAB->JAjb', imds.Wovvo, IAB).reshape(nkpts,nkpts,nocc,1,nvir) +\
                    einsum('JAAjaaj->JAja', imds.Wovvo).reshape(nkpts,nkpts,nocc,nvir,1) +\
                    einsum('ABAabab,JAB->JAab', imds.Wvvvv, IAB).reshape(nkpts,nkpts,1,nvir,nvir) -\
                    einsum('JAkjab,JAkjab->JAjab', imds.Woovv[kshift], imds.t2[kshift]))

        Hr2 = array(Hr2, sym2)
        return self.amplitudes_to_vector_ea(Hr1, Hr2)

    def ccsd_t(self, t1=None, t2=None, eris=None, slice_size=None):
        from pyscf.ctfcc import kccsd_t
        return kccsd_t.kernel(self, eris, t1, t2, slice_size)

    def ccsd_t_slow(self, t1=None, t2=None, eris=None):
        raise NotImplementedError

KGCCSD = GCCCSD = KCCSD

class _PhysicistsERIs:
    def __init__(self, mycc, mo_coeff=None):
        from pyscf.pbc.cc.ccsd import _adjust_occ
        from pyscf.pbc import df
        cput0 = (time.clock(), time.time())
        nocc, nmo, nkpts = mycc.nocc, mycc.nmo, mycc.nkpts
        nvir = nmo - nocc
        cell, kpts = mycc._scf.cell, mycc.kpts
        nao = cell.nao_nr()
        sym2 = mycc.gen_sym('+-+-')
        nonzero_opadding, nonzero_vpadding = padding_k_idx(mycc, kind="split")
        if mo_coeff is None: mo_coeff = mycc._scf.mo_coeff
        nao = mo_coeff[0].shape[0]
        dtype = mo_coeff[0].dtype
        moidx = mycc.get_frozen_mask()
        nocc_per_kpt = np.asarray(mycc.get_nocc(per_kpoint=True))
        nmo_per_kpt  = np.asarray(mycc.get_nmo(per_kpoint=True))

        padded_moidx = []
        for k in range(nkpts):
            kpt_nocc = nocc_per_kpt[k]
            kpt_nvir = nmo_per_kpt[k] - kpt_nocc
            kpt_padded_moidx = np.concatenate((np.ones(kpt_nocc, dtype=np.bool),
                                                  np.zeros(nmo - kpt_nocc - kpt_nvir, dtype=np.bool),
                                                  np.ones(kpt_nvir, dtype=np.bool)))
            padded_moidx.append(kpt_padded_moidx)

        self.mo_coeff = []
        self.orbspin = []
        self.kconserv = mycc.khelper.kconserv
        # Generate the molecular orbital coefficients with the frozen orbitals masked.
        # Each MO is tagged with orbspin, a list of 0's and 1's that give the overall
        # spin of each MO.
        #
        # Here we will work with two index arrays; one is for our original (small) moidx
        # array while the next is for our new (large) padded array.
        for k in range(nkpts):
            kpt_moidx = moidx[k]
            kpt_padded_moidx = padded_moidx[k]

            mo = np.zeros((nao, nmo), dtype=dtype)
            mo[:, kpt_padded_moidx] = mo_coeff[k][:, kpt_moidx]
            if getattr(mo_coeff[k], 'orbspin', None) is not None:
                orbspin_dtype = mo_coeff[k].orbspin[kpt_moidx].dtype
                orbspin = np.zeros(nmo, dtype=orbspin_dtype)
                orbspin[kpt_padded_moidx] = mo_coeff[k].orbspin[kpt_moidx]
                mo = lib.tag_array(mo, orbspin=orbspin)
                self.orbspin.append(orbspin)
            else:  # guess orbital spin - assumes an RHF calculation
                assert (np.count_nonzero(kpt_moidx) % 2 == 0)
                orbspin = np.zeros(mo.shape[1], dtype=int)
                orbspin[1::2] = 1
                mo = lib.tag_array(mo, orbspin=orbspin)
                self.orbspin.append(orbspin)
            self.mo_coeff.append(mo)
        # Re-make our fock MO matrix elements from density and fock AO
        if rank==0:
            dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
            with lib.temporary_env(mycc._scf, exxdiv=None):
                # _scf.exxdiv affects eris.fock. HF exchange correction should be
                # excluded from the Fock matrix.
                vhf = mycc._scf.get_veff(cell, dm)
            fockao = mycc._scf.get_hcore() + vhf
            self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                       for k, mo in enumerate(self.mo_coeff)])
            self.e_hf = mycc._scf.energy_tot(dm=dm, vhf=vhf)
            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
        else:
            self.fock = self.e_hf = self.mo_energy = None

        ctf_helper.synchronize(self, ["fock", "e_hf", "mo_energy", "mo_coeff", "orbspin"])
        # Add HFX correction in the eris.mo_energy to improve convergence in
        # CCSD iteration. It is useful for the 2D systems since their occupied and
        # the virtual orbital energies may overlap which may lead to numerical
        # issue in the CCSD iterations.
        # FIXME: Whether to add this correction for other exxdiv treatments?
        # Without the correction, MP2 energy may be largely off the correct value.
        madelung = tools.madelung(cell, kpts)
        self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                          for k, mo_e in enumerate(self.mo_energy)]

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = mycc.get_nocc(per_kpoint=True)
        nonzero_padding = padding_k_idx(mycc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)
        sym1, sym2 = mycc._sym[:2]
        fock = asarray(self.fock)
        self.foo = array(fock[:,:nocc,:nocc], sym1)
        self.fov = array(fock[:,:nocc,nocc:], sym1)
        self.fvv = array(fock[:,nocc:,nocc:], sym1)
        mo_e_o = [e[:nocc] for e in self.mo_energy]
        mo_e_v = [e[nocc:] + mycc.level_shift for e in self.mo_energy]
        eia = np.zeros([nkpts,nocc,nvir])
        for ki in range(nkpts):
            eia[ki] = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                        [0,nvir,ki,mo_e_v,nonzero_vpadding],
                        fac=[1.0,-1.0])

        self.eia = asarray(eia)
        self._foo = asarray([np.diag(e) for e in mo_e_o])
        self._fvv = asarray([np.diag(e) for e in mo_e_v])
        kconserv = mycc.khelper.kconserv

        all_tasks = [[ki,kj,ka] for ki,kj,ka in itertools.product(range(nkpts), repeat=3)]
        script_mo = (nocc, nvir, mo_e_o, mo_e_v, nonzero_opadding, nonzero_vpadding)
        get_eijab  = lambda ki,kj,ka: kccsd_rhf._get_eijab(ki, kj, ka, kconserv,script_mo)
        self.eijab = frombatchfunc(get_eijab, (nocc,nocc,nvir,nvir), all_tasks, sym=sym2)

        if type(mycc._scf.with_df) is df.FFTDF:
            ao2mo.make_fftdf_eris_ghf(mycc, self)
        else:
            from pyscf.ctfcc.integrals import mpigdf
            if type(mycc._scf.with_df) is mpigdf.GDF:
                ao2mo.make_df_eris_ghf(mycc, self)
            elif type(mycc._scf.with_df) is df.GDF:
                logger.warn(mycc, "GDF converted to an MPIGDF object, \
                                   one process used for reading from disk")
                mycc._scf.with_df = mpigdf.from_serial(mycc._scf.with_df)
                ao2mo.make_df_eris_ghf(mycc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        logger.timer(mycc, "ao2mo transformation", *cput0)

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
    kmf = scf.KRHF(cell, kpts=cell.make_kpts([1, 1, 3]), exxdiv=None)
    kmf.kernel()
    kmf = kmf.to_ghf(kmf)

    mycc = KGCCSD(kmf)
    e, t1, t2 = mycc.kernel()
    print(e- -0.01031588020568685)

    eip, vip = mycc.ipccsd(nroots=3, kptlist=[1])
    eea, vea = mycc.eaccsd(nroots=3, kptlist=[2])

    print(eip[0,0] - 0.1344891429406715)
    print(eip[0,1] - 0.1344891483753267)
    print(eip[0,2] - 0.4827325097125353)
    print(eea[0,0] - 1.609383466425309)
    print(eea[0,1] - 1.609383469756525)
    print(eea[0,2] - 2.228400562950027)
