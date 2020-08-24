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
RCCSD with CTF as backend, see pyscf.cc.rccsd_slow
'''
import numpy as np
import time
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import rccsd_slow, ccsd

from pyscf.ctfcc.integrals.ao2mo import make_ao_ints
from pyscf.ctfcc.linalg_helper import davidson
from pyscf.ctfcc.linalg_helper.diis import DIIS
from pyscf.ctfcc.ctf_helper import pack_tril, unpack_tril, synchronize
from symtensor.ctf import einsum, diag, array, frombatchfunc
from symtensor.ctf.backend import hstack, dot, asarray, eye, argsort, norm

rccsd_slow.imd.einsum = rccsd_slow.einsum = einsum
rccsd_slow.dot = dot
rccsd_slow.asarray = asarray
rccsd_slow.eye = eye

def kernel(mycc, eris=None, t1=None, t2=None, max_cycle=50, tol=1e-8,
           tolnormt=1e-6, verbose=None):
    '''
    ctf version of pyscf.cc.ccsd.kernel
    '''
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    if t1 is None: t1, t2 = mycc.init_amps(eris)[1:]

    if isinstance(mycc.diis, DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = DIIS(mycc)
        adiis.space = mycc.diis_space
    else:
        adiis = None
    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    eccsd = 0
    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = mycc.get_normt_diff(t1, t2, t1new, t2new)
        if mycc.iterative_damping < 1.0:
            alpha = mycc.iterative_damping
            t1new = (1-alpha) * t1 + alpha * t1new
            t2new *= alpha
            t2new += (1-alpha) * t2
        t1, t2 = t1new, t2new
        t1new = t2new = None
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        logger.info(mycc, 'cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = logger.timer(mycc, 'CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    logger.timer(mycc, 'CCSD', *cput0)

    return conv, eccsd, t1, t2

def lambda_kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO,
           fintermediates=None, fupdate=None):
    '''
    ctf kernel for solving lambda equations
    '''
    if eris is None: eris = mycc.ao2mo()
    cput0 = (time.clock(), time.time())
    log = logger.new_logger(mycc, verbose)

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = t1
    if l2 is None: l2 = t2
    if fintermediates is None:
        fintermediates = make_intermediates
    if fupdate is None:
        fupdate = update_lambda

    imds = fintermediates(mycc, t1, t2, eris)

    if isinstance(mycc.diis, DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = DIIS(mycc)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    cput0 = log.timer('CCSD lambda initialization', *cput0)

    conv = False
    for istep in range(max_cycle):
        l1new, l2new = fupdate(mycc, t1, t2, l1, l2, eris, imds)
        normt = mycc.get_normt_diff(l1, l2, l1new, l2new)
        l1, l2 = l1new, l2new
        l1new = l2new = None
        l1, l2 = mycc.run_diis(l1, l2, istep, normt, 0, adiis)
        log.info('cycle = %d  norm(lambda1,lambda2) = %.6g', istep+1, normt)
        cput0 = log.timer('CCSD iter', *cput0)
        if normt < tol:
            conv = True
            break
    return conv, l1, l2

def get_normt_diff(mycc, t1, t2, t1new, t2new):
    '''compute norm difference for two sets of amplitudes'''
    deltavec = mycc.amplitudes_to_vector(t1new, t2new) - \
               mycc.amplitudes_to_vector(t1, t2)
    return norm(deltavec)

def amplitudes_to_vector(t1, t2):
    nocc, nvir = t1.shape
    nov = nocc*nvir
    t2v = t2.transpose(0,2,1,3).reshape(nov, nov)
    t2v = pack_tril(t2v)
    vector = hstack((t1.ravel(), t2v))
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].reshape(nocc,nvir)
    t2 = unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC).reshape(nocc,nvir,nocc,nvir)
    t2 = t2.transpose(0,2,1,3)
    return t1, t2

def amplitudes_to_vector_ip(r1, r2):
    return hstack((r1, r2.ravel()))

def amplitudes_to_vector_ea(r1, r2):
    return hstack((r1, r2.ravel()))

class RCCSD(rccsd_slow.RCCSD):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        synchronize(mf, ['mo_coeff', 'mo_energy', 'mo_occ'])
        rccsd_slow.RCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

    get_normt_diff = get_normt_diff

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
            cc2 : bool
                Use CC2 approximation to CCSD.
        '''
        if mbpt2 and cc2:
            raise RuntimeError('MBPT2 and CC2 are mutually exclusive approximations to the CCSD ground state.')
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.eris = eris
        self.dump_flags()
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            if cc2:
                cctyp = 'CC2'
                self.cc2 = True
            else:
                cctyp = 'CCSD'
                self.cc2 = False
            self.converged, self.e_corr, self.t1, self.t2 = \
                    kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                                tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                                verbose=self.verbose)
            if self.converged:
                logger.info(self, '%s converged', cctyp)
            else:
                logger.info(self, '%s not converged', cctyp)
        if self._scf.e_tot == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def amplitudes_to_vector(self, t1, t2):
        return amplitudes_to_vector(t1, t2)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                lambda_kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose,
                                    fintermediates=rccsd_slow.make_intermediates,
                                    fupdate=rccsd_slow.update_lambda)
        return self.l1, self.l2

    def get_init_guess_ip(self, nroots=1, koopmans=False, diag=None):
        size = self.nip()
        nocc = self.nocc
        if koopmans:
            idx = range(nocc-nroots, nocc)[::-1]
        else:
            if diag is None:
                diag = self.ipccsd_diag()
            idx = argsort(diag)[:nroots]
        def write_guess(i):
            return i*size+idx[i], np.ones(1)
        all_tasks = np.arange(nroots)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, all_tasks).array
        return guess

    def get_init_guess_ea(self, nroots=1, koopmans=False, diag=None):
        size = self.nea()
        if koopmans:
            idx = range(nroots)
        else:
            if diag is None:
                diag = self.eaccsd_diag()
            idx = argsort(diag)[:nroots]
        def write_guess(i):
            return i*size+idx[i], np.ones(1)
        all_tasks = np.arange(nroots)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, all_tasks).array
        return guess

    def get_init_guess_ee(self, nroots=1, koopmans=True, diag=None):
        if diag is None:
            diag = self.eeccsd_diag()
        size = self.nee()
        nroots = min(nroots, size)
        if koopmans:
            nocc = self.nocc
            nvir = self.nmo - nocc
            idx = argsort(diag[:nocc*nvir])[:nroots]
        else:
            idx = argsort(diag)[:nroots]
        dtype = getattr(diag, 'dtype', np.double)
        def write_guess(i):
            return i*size+idx[i], np.ones(1)
        all_tasks = np.arange(nroots)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, all_tasks).array
        return guess

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        cput0 = (time.clock(), time.time())
        diag = self.ipccsd_diag()
        log = logger.Logger(self.stdout, self.verbose)
        if left:
            matvec = lambda x: self.lipccsd_matvec(x)
        else:
            matvec = lambda x: self.ipccsd_matvec(x)
        size = self.nip()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ip_partition = partition
        if partition == 'full':
            self._ipccsd_diag_matrix2 = self.vector_to_amplitudes_ip(self.ipccsd_diag())[1]

        if guess is None:
            user_guess = False
            guess = self.get_init_guess_ip(nroots, koopmans, diag)
        else:
            if isinstance(guess, (tuple,list)):
                for g in guess:
                    assert(g.size==size)
            else:
                assert(guess.shape[-1]==size)
            user_guess = True

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = davidson.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = davidson._gen_x0(envs['v'], envs['xs'])
                s = einsum('ab,cb->ac', guess.conj(), x0)
                snorm = einsum('pi,pi->i', s.conj(), s).to_nparray()
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

        for kx, (n, en) in enumerate(zip(range(nroots), eip)):
            r1, r2 = self.vector_to_amplitudes_ip(evecs[kx])
            if isinstance(r1, (tuple, list)):
                qp_weight = sum([norm(ri)**2 for ri in r1])
            else:
                qp_weight = norm(r1)**2
            logger.info(self, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)

        log.timer('IP-CCSD', *cput0)
        if nroots == 1:
            return eip[0], evecs[0]
        else:
            return eip, evecs

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None, partition=None):
        cput0 = (time.clock(), time.time())
        diag = self.eaccsd_diag()
        log = logger.Logger(self.stdout, self.verbose)
        if left:
            matvec = lambda x: self.leaccsd_matvec(x)
        else:
            matvec = lambda x: self.eaccsd_matvec(x)
        size = self.nea()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)

        if partition:
            partition = partition.lower()
            assert partition in ['mp','full']
        self.ea_partition = partition
        if partition == 'full':
            self._eaccsd_diag_matrix2 = self.vector_to_amplitudes_ea(self.eaccsd_diag())[1]

        if guess is None:
            user_guess = False
            guess = self.get_init_guess_ea(nroots, koopmans, diag)
        else:
            if isinstance(guess, (tuple,list)):
                for g in guess:
                    assert(g.size==size)
            else:
                assert(guess.shape[-1]==size)
            user_guess = True

        def precond(r, e0, x0):
            return r/(e0-diag+1e-12)

        eig = davidson.eig
        if user_guess or koopmans:
            def pickeig(w, v, nr, envs):
                x0 = davidson._gen_x0(envs['v'], envs['xs'])
                s = einsum('ab,cb->ac', guess.conj(), x0)
                snorm = einsum('pi,pi->i', s.conj(), s).to_nparray()
                idx = np.argsort(-snorm)[:nroots]
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

        for kx, (n, en) in enumerate(zip(range(nroots), eea)):
            r1, r2 = self.vector_to_amplitudes_ea(evecs[kx])
            if isinstance(r1, (tuple, list)):
                qp_weight = sum([norm(ri)**2 for ri in r1])
            else:
                qp_weight = norm(r1)**2
            logger.info(self, 'EOM-CCSD root %d E = %.16g  qpwt = %0.6g', n, en, qp_weight)

        log.timer('EA-CCSD', *cput0)
        if nroots == 1:
            return eea[0], evecs[0]
        else:
            return eea, evecs

    def amplitudes_to_vector_ip(self, r1, r2):
        return amplitudes_to_vector_ip(r1, r2)

    def amplitudes_to_vector_ea(self, r1, r2):
        return amplitudes_to_vector_ea(r1, r2)

class _ChemistsERIs(rccsd_slow._ChemistsERIs):
    def __init__(self, cc, mo_coeff=None):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))
        nocc = cc.nocc
        nvir = cc.nmo - nocc
        mo_e = fock.diagonal().real
        self.eia = asarray(mo_e[:nocc,None] - mo_e[None,nocc:])
        self.eijab = self.eia.reshape(1,nocc,1,nvir) + self.eia.reshape(nocc,1,nvir,1)
        fock = asarray(fock)
        self.foo = fock[:nocc,:nocc]
        self.fov = fock[:nocc,nocc:]
        self.fvv = fock[nocc:,nocc:]
        self._foo = diag(diag(self.foo))
        self._fvv = diag(diag(self.fvv))

        dtype = fock.dtype
        cput0 = cput1 = (time.clock(), time.time())
        ppoo, ppov, ppvv = make_ao_ints(cc.mol, mo_coeff, nocc)
        cput1 = logger.timer(cc, 'making ao integrals', *cput1)
        mo = asarray(mo_coeff)
        orbo, orbv = mo[:,:nocc], mo[:,nocc:]

        tmp = einsum('uvmn,ui->ivmn', ppoo, orbo.conj())
        self.oooo = einsum('ivmn,vj->ijmn', tmp, orbo)
        self.ooov = einsum('ivmn,va->mnia', tmp, orbv)
        tmp = einsum('uvma,vb->ubma', ppov, orbv)
        self.ovov = einsum('ubma,ui->ibma', tmp, orbo.conj())
        tmp = einsum('uvma,ub->mabv', ppov, orbv.conj())
        self.ovvo = einsum('mabv,vi->mabi', tmp, orbo)

        tmp = einsum('uvab,ui->ivab', ppvv, orbo.conj())
        self.oovv = einsum('ivab,vj->ijab', tmp, orbo)

        tmp = einsum('uvab,vc->ucab', ppvv, orbv)
        self.ovvv = einsum('ucab,ui->icab', tmp, orbo.conj())
        self.vvvv = einsum('ucab,ud->dcab', tmp, orbv.conj())
        logger.timer(cc, 'ao2mo transformation', *cput0)

if __name__ == '__main__':
    from pyscf import gto, scf, cc
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run(conv_tol=1e-14)

    mycc = RCCSD(mf)
    eris = mycc.ao2mo()
    e, t1, t2 = mycc.kernel(eris=eris)
    print(e - -0.2133432467414933)


    et = mycc.ccsd_t(t1, t2, eris)
    et1 = mycc.ccsd_t_slow(t1, t2, eris)

    print(et - -0.0030600233005741453)
    print(et1 - -0.0030600233005741453)
    mycc.verbose = 5
    t0 = time.time()
    eip, vip = mycc.ipccsd1(nroots=2, left=True)
    print(eip[0] - 0.43356041)
    print(eip[1] - 0.51876597)
    t1=time.time()

    eip, vip = mycc.ipccsd(nroots=2, left=True)
    print(eip[0] - 0.43356041)
    print(eip[1] - 0.51876597)
    t2= time.time()
    print(t1-t0,t2-t1)


    eea, vea = mycc.eaccsd(nroots=2, left=True)
    print(eea[0] - 0.16737886)
    print(eea[1] - 0.24027623)

    eip, vip = mycc.ipccsd_t_star(nroots=2, left=True)
    print(eip[0] - 0.43455702)
    print(eip[1] - 0.51991415)

    eea, vea = mycc.eaccsd_t_star(nroots=2, left=True)
    print(eea[0] - 0.16785694)
    print(eea[1] - 0.24098345)
