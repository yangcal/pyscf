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
GCCSD with CTF as backend, see pyscf.cc.gccsd_slow
'''

import numpy as np
import time
from pyscf.lib import logger
from pyscf.cc import gccsd_slow

from pyscf.ctfcc import rccsd
from pyscf.ctfcc.integrals.ao2mo import make_ao_ints
from pyscf.ctfcc import ctf_helper
from symtensor.ctf import einsum
from symtensor.ctf.backend import hstack, asarray, norm, diag, dot

gccsd_slow.imd.einsum = gccsd_slow.einsum = einsum
gccsd_slow.asarray = asarray
gccsd_slow.dot = dot

def amplitudes_to_vector_ip(r1, r2):
    r2v = ctf_helper.pack_ip_r2(r2)
    return hstack((r1, r2v))

def vector_to_amplitudes_ip(vector, nmo, nocc):
    r1 = vector[:nocc]
    r2 = ctf_helper.unpack_ip_r2(vector[nocc:], nmo, nocc)
    return r1, r2

def amplitudes_to_vector_ea(r1, r2):
    r2v = ctf_helper.pack_ea_r2(r2)
    return hstack((r1, r2v))

def vector_to_amplitudes_ea(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nvir]
    r2 = ctf_helper.unpack_ea_r2(vector[nvir:], nmo, nocc)
    return r1, r2

class GCCSD(gccsd_slow.GCCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ctf_helper.synchronize(mf, ['mo_coeff', 'mo_energy', 'mo_occ'])
        gccsd_slow.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    get_normt_diff = rccsd.RCCSD.get_normt_diff
    ccsd = rccsd.RCCSD.ccsd
    ipccsd = rccsd.RCCSD.ipccsd
    eaccsd = rccsd.RCCSD.eaccsd
    get_init_guess_ip = rccsd.RCCSD.get_init_guess_ip
    get_init_guess_ea = rccsd.RCCSD.get_init_guess_ea
    get_init_guess_ee = rccsd.RCCSD.get_init_guess_ee

    def amplitudes_to_vector(self, t1, t2):
        return ctf_helper.amplitudes_to_vector_s4(t1, t2)

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return ctf_helper.vector_to_amplitudes_s4(vector, nmo, nocc)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                rccsd.lambda_kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose,
                                    fintermediates=gccsd_slow.make_intermediates,
                                    fupdate=gccsd_slow.update_lambda)
        return self.l1, self.l2

    amplitudes_to_vector_ee = amplitudes_to_vector
    vector_to_amplitudes_ee = vector_to_amplitudes

    def amplitudes_to_vector_ip(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def vector_to_amplitudes_ip(self, vector, **kwargs):
        nocc = self.nocc
        nmo = self.nmo
        return vector_to_amplitudes_ip(vector, nmo, nocc)

    def amplitudes_to_vector_ea(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

    def vector_to_amplitudes_ea(self, vector, **kwargs):
        nocc = self.nocc
        nmo = self.nmo
        return vector_to_amplitudes_ea(vector, nmo, nocc)

    def eeccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        cput0 = (time.clock(), time.time())
        diag = self.eeccsd_diag()
        log = logger.Logger(self.stdout, self.verbose)
        if left:
            matvec = lambda x: self.leeccsd_matvec(x)
        else:
            matvec = lambda x: self.eeccsd_matvec(x)
        size = self.nee()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)

        if guess is None:
            guess = self.get_init_guess_ee(nroots, koopmans, diag)
        else:
            if isinstance(guess, (tuple,list)):
                for g in guess:
                    assert(g.size==size)
            else:
                assert(guess.shape[-1]==size)

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        from pyscf.ctfcc.linalg_helper.davidson import eig
        eee, evecs = eig(matvec, guess, precond,
                         tol=self.conv_tol, max_cycle=self.max_cycle,
                         max_space=self.max_space, nroots=nroots, verbose=self.verbose)

        self.eee = eee
        if nroots == 1:
            eee, evecs = [self.eee], [evecs]

        for kx, (n, en) in enumerate(zip(range(nroots), eee)):
            r1, r2 = self.vector_to_amplitudes_ee(evecs[kx])
            if isinstance(r1, (tuple, list)):
                qp_weight = sum([norm(ri)**2 for ri in r1])
            else:
                qp_weight = norm(r1)**2
            logger.info(self, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)

        log.timer('EE-CCSD', *cput0)
        if nroots == 1:
            return eee[0], evecs[0]
        else:
            return eee, evecs


def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = gccsd_slow._PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    eris.fock = asarray(eris.fock)
    eris.foo = eris.fock[:nocc,:nocc]
    eris.fov = eris.fock[:nocc,nocc:]
    eris.fvv = eris.fock[nocc:,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eris._foo  = diag(diag(eris.foo))
    eris._fvv  = diag(diag(eris.fvv))
    eris.eia = mo_e_o[:,None] - mo_e_v
    eris.eia = asarray(eris.eia)
    eris.eijab = eris.eia.reshape(nocc,1,nvir,1) + eris.eia.reshape(1,nocc,1,nvir)

    nao = eris.mo_coeff.shape[0]
    mo_a = eris.mo_coeff[:nao//2]
    mo_b = eris.mo_coeff[nao//2:]

    cput1 = cput0 = (time.clock(), time.time())
    ppoo, ppov, ppvv = make_ao_ints(mycc.mol, mo_a+mo_b, nocc)
    orbspin = eris.orbspin
    occspin = orbspin[:nocc]
    virspin = orbspin[nocc:]

    oo = np.ones([nocc,nocc]) # delta tensors to force orbital symmetry
    ov = np.ones([nocc,nvir])
    vv = np.ones([nvir,nvir])

    oo[occspin[:,None]!=occspin] = 0
    ov[occspin[:,None]!=virspin] = 0
    vv[virspin[:,None]!=virspin] = 0

    oo = asarray(oo)
    ov = asarray(ov)
    vv = asarray(vv)

    ppoo = einsum('uvmn,mn->uvmn', ppoo, oo)
    ppov = einsum('uvma,ma->uvma', ppov, ov)
    ppvv = einsum('uvab,ab->uvab', ppvv, vv)

    cput1 = logger.timer(mycc, 'making ao integrals', *cput1)
    mo = asarray(mo_a+mo_b)
    orbo, orbv = mo[:,:nocc], mo[:,nocc:]

    tmp = einsum('uvmn,ui->ivmn', ppoo, orbo)
    oooo = einsum('ivmn,vj,ij->ijmn', tmp, orbo, oo)

    eris.oooo = oooo.transpose(0,2,1,3) - oooo.transpose(0,2,3,1)
    ooov = einsum('ivmn,va,ia->mnia', tmp, orbv,ov)
    eris.ooov = ooov.transpose(0,2,1,3) - ooov.transpose(2,0,1,3)

    tmp = einsum('uvma,vb->ubma', ppov, orbv)
    ovov = einsum('ubma,ui,ib->ibma', tmp, orbo, ov)
    eris.oovv = ovov.transpose(0,2,1,3) - ovov.transpose(0,2,3,1)
    del ppoo, ovov, tmp

    tmp = einsum('uvma,ub->mabv', ppov, orbv)
    _ovvo = einsum('mabv,vi,ib->mabi', tmp, orbo, ov)
    tmp = einsum('uvab,ui->ivab', ppvv, orbo)
    _oovv = einsum('ivab,vj,ij->ijab', tmp, orbo,oo)

    eris.ovov = _oovv.transpose(0,2,1,3) - _ovvo.transpose(0,2,3,1)
    eris.ovvo = _ovvo.transpose(0,2,1,3) - _oovv.transpose(0,2,3,1)
    del _ovvo, _oovv, ppov, tmp

    tmp = einsum('uvab,vc->ucab', ppvv, orbv)
    ovvv = einsum('ucab,ui,ic->icab', tmp, orbo, ov)
    eris.ovvv = ovvv.transpose(0,2,1,3) - ovvv.transpose(0,2,3,1)
    vvvv = einsum('ucab,ud,dc->dcab', tmp, orbv, vv)
    eris.vvvv = vvvv.transpose(0,2,1,3) - vvvv.transpose(0,2,3,1)
    del ovvv, vvvv, ppvv, tmp
    logger.timer(mycc, 'ao2mo transformation', *cput0)
    return eris

gccsd_slow._make_eris_incore = _make_eris_incore

if __name__ == '__main__':
    from pyscf import gto, scf

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

    e,veal = mycc.eaccsd(nroots=8, left=True)
    print(e[0] - 0.16737886338859731)
    print(e[2] - 0.24027613852009164)

    estar = mycc.eaccsd_star_contract(e, vear, veal)
    print(estar[0] - 0.16656253472780994)
    print(estar[2] - 0.23944154865211192)

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
