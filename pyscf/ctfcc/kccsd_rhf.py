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
KRCCSD with CTF as backend, all integrals in memory
'''
import numpy as np
import time
from functools import reduce
import itertools
from pyscf import lib
from pyscf.lib import logger
from pyscf import __config__
from pyscf.cc import rccsd_slow
from pyscf.pbc.mp.kmp2 import (padded_mo_coeff, padding_k_idx)
from pyscf.pbc.cc import kccsd_rhf, eom_kccsd_rhf
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.ctfcc import rccsd
from pyscf.ctfcc.integrals import ao2mo
from pyscf.ctfcc import rintermediates as imd
from symtensor.ctf import einsum, array, frombatchfunc, zeros
from symtensor.ctf.backend import hstack, asarray, eye, argsort, vstack
from symtensor.symlib import SYMLIB
from pyscf import __config__

from pyscf.ctfcc import ctf_helper
rank = ctf_helper.rank

rccsd_slow.imd.get_t3p2_imds_slow = imd.get_t3p2_imds_slow
SLICE_SIZE = getattr(__config__, 'ctfcc_kccsd_slice_size', 4000)


def energy(mycc, t1, t2, eris):
    nkpts = mycc.nkpts
    e = 2*einsum('ia,ia', eris.fov, t1)
    tau = einsum('ia,jb->ijab',t1,t1)
    tau += t2
    e += 2*einsum('ijab,iajb', tau, eris.ovov)
    e +=  -einsum('ijab,ibja', tau, eris.ovov)
    if abs(e.imag)>1e-4:
        logger.warn(mycc, 'Non-zero imaginary part found in KRCCSD energy %s', e)
    return e.real / nkpts

def init_amps(mycc, eris):
    time0 = time.clock(), time.time()
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    sym1 = mycc._sym[0]
    t1 = zeros([nocc,nvir], sym1, eris.dtype)
    t2 = eris.ovov.transpose(0,2,1,3).conj() / eris.eijab
    mycc.emp2  = 2*einsum('ijab,iajb', t2, eris.ovov)
    mycc.emp2 -=   einsum('ijab,ibja', t2, eris.ovov)
    mycc.emp2  =   mycc.emp2.real/ mycc.nkpts
    logger.info(mycc, 'Init t2, MP2 energy (with fock eigenvalue shift) = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, t1, t2

def amplitudes_to_vector(t1, t2):
    vector = hstack((t1.ravel(), t2.ravel()))
    return vector

class KRCCSD(kccsd_rhf.KRCCSD):
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, slice_size=SLICE_SIZE):
        ctf_helper.synchronize(mf, ["mo_coeff", "mo_occ", "mo_energy"])
        kccsd_rhf.KRCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self.ip_partition = self.ea_partition = None
        self.slice_size = SLICE_SIZE
        self.max_space = getattr(__config__, 'pbc_cc_kccsd_rhf_KRCCSD_max_space', 20)
        self.symlib = SYMLIB('ctf')
        self.__imds__  = None
        self._keys = self._keys.union(['max_space', 'ip_partition', '__imds__'\
                                       'ea_partition', 'symlib', 'slice_size'])
        self.make_symlib()

    init_amps = init_amps
    energy = energy

    update_amps = rccsd.RCCSD.update_amps
    ccsd = rccsd.RCCSD.ccsd
    kernel = rccsd.RCCSD.kernel

    ipccsd_matvec = rccsd.RCCSD.ipccsd_matvec
    eaccsd_matvec = rccsd.RCCSD.eaccsd_matvec
    lipccsd_matvec = rccsd.RCCSD.lipccsd_matvec
    leaccsd_matvec = rccsd.RCCSD.leaccsd_matvec

    nip = eom_kccsd_rhf.EOMIP.vector_size
    nea = eom_kccsd_rhf.EOMEA.vector_size

    @property
    def _sym(self):
        '''Descriptors of Kpoint symmetry in T1/T2'''
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym1 = self.gen_sym('+-')
        sym2 = self.gen_sym('++--')
        symq = ['++-',[kpts,kpts,kpts[0]+kpts], None, gvec]
        return sym1, sym2, symq

    def make_symlib(self):
        '''Pre-compute all transformation deltas needed in KCCSD iterations'''
        self.symlib.update(*self._sym)

    def get_normt_diff(self, t1, t2, t1new, t2new):
        return (t1new-t1).norm() + (t2new - t2).norm()

    def vector_to_amplitudes(self, vector):
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nmo - self.nocc
        sym1, sym2 = self._sym[:2]
        size_t1 = nkpts*nocc*nvir
        t1 = array(vector[:size_t1].reshape(nkpts,nocc,nvir), sym=sym1)
        t2 = array(vector[size_t1:].reshape(nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), sym=sym2)
        t1.symlib = t2.symlib = self.symlib
        return t1, t2

    def amplitudes_to_vector(self, t1, t2, **kwargs):
        return amplitudes_to_vector(t1, t2)

    amplitudes_to_vector_ip = amplitudes_to_vector_ea = amplitudes_to_vector

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)

    @property
    def imds(self):
        if self.__imds__ is None:
            self.__imds__ = rccsd_slow._IMDS(self)
        return self.__imds__

    def get_init_guess_ip(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.nip()
        nroots = min(nroots, size)
        nonzero_opadding = padding_k_idx(self, kind="split")[0]
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(nonzero_opadding[kshift][::-1][:nroots])]
        else:
            if diag is None:
                diag = self.ipccsd_diag(kshift)
            idx = argsort(diag)[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        def write_guess(i):
            return ind[i], np.ones(1)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, np.arange(nroots)).array
        return guess

    def get_init_guess_ea(self, kshift, nroots=1, koopmans=False, diag=None):
        size = self.nea()
        nroots = min(nroots, size)
        nonzero_vpadding = padding_k_idx(self, kind="split")[1]
        if koopmans:
            ind = [kn*size+n for kn, n in enumerate(nonzero_vpadding[kshift][:nroots])]
        else:
            if diag is None:
                diag = self.eaccsd_diag(kshift)
            idx = argsort(diag)[:nroots]
            ind = [kn*size+n for kn, n in enumerate(idx)]
        def write_guess(i):
            return ind[i], np.ones(1)
        shape = (nroots, size)
        guess = frombatchfunc(write_guess, shape, np.arange(nroots)).array
        return guess



    def ipccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        cput0 = (time.clock(), time.time())

        from pyscf.ctfcc.linalg_helper.davidson import davidson_nosym1
        size = self.nip()
        nroots = min(nroots,size)
        if kptlist is None:
            kptlist = range(nkpts)

        if guess is not None:
            user_guess = True
            if isinstance(guess, (tuple,list)):
                for g in guess:
                    assert(g.size==size)
            else:
                assert(guess.shape==(nroots,size))
        else:
            user_guess = False

        evals = np.zeros([len(kptlist),nroots])
        evecs = [None] * len(kptlist)
        convs = np.zeros((len(kptlist),nroots))
        for k, kshift in enumerate(kptlist):
            diag = self.ipccsd_diag(kshift)
            if guess is None:
                g = self.get_init_guess_ip(kshift, nroots, koopmans, diag)
            else:
                g = guess

            if left:
                matvec = lambda x: self.lipccsd_matvec(x, kshift=kshift)
            else:
                matvec = lambda x: self.ipccsd_matvec(x, kshift=kshift)

            def precond(r, e0, x0):
                return r/(e0-diag+1e-12)

            def gen_matvec(xs):
                if isinstance(xs, (tuple, list)):
                    nvec = len(xs)
                else:
                    nvec = xs.shape[0]
                out = [matvec(xs[i]) for i in range(nvec)]
                return out
            conv_k, evals_k, evecs_k = davidson_nosym1(gen_matvec, g, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)
            evals[k] = evals_k
            evecs[k] = evecs_k
            convs[k] = conv_k
            if nroots==1:
                evals_k = [evals_k]
                evecs_k = [evecs_k]
            for n in range(nroots):
                r1, r2 = self.vector_to_amplitudes_ip(evecs_k[n], kshift=kshift)
                if isinstance(r1, (tuple, list)):
                    qp_weight = sum([ri.norm()**2 for ri in r1])
                else:
                    qp_weight = r1.norm() **2
                logger.info(self, 'EOM-IP-CCSD root %d E = %.16g  qpwt = %0.6g',
                            n, evals_k[n], qp_weight)
        evecs = vstack(tuple(evecs)).reshape(len(kptlist),nroots,size)
        logger.timer(self, 'EOM-CCSD', *cput0)
        return evals, evecs


    def eaccsd(self, nroots=1, koopmans=True, guess=None, left=False,
               eris=None, imds=None, partition=None, kptlist=None,
               dtype=None, **kwargs):
        from pyscf.ctfcc.linalg_helper.davidson import davidson_nosym1
        cput0 = (time.clock(), time.time())
        size = self.nea()
        nroots = min(nroots,size)
        if kptlist is None:
            kptlist = range(nkpts)

        if guess is not None:
            if isinstance(guess, (tuple,list)):
                for g in guess:
                    assert(g.size==size)
            else:
                assert(guess.shape==(nroots,size))

        evals = np.zeros([len(kptlist),nroots])
        evecs = [None] * len(kptlist)
        convs = np.zeros((len(kptlist),nroots))
        for k, kshift in enumerate(kptlist):
            diag = self.eaccsd_diag(kshift)
            if guess is None:
                g = self.get_init_guess_ea(kshift, nroots, koopmans, diag)
            else:
                g = guess

            if left:
                matvec = lambda x: self.leaccsd_matvec(x, kshift=kshift)
            else:
                matvec = lambda x: self.eaccsd_matvec(x, kshift=kshift)

            def precond(r, e0, x0):
                return r/(e0-diag+1e-12)

            def gen_matvec(xs):
                if isinstance(xs, (tuple, list)):
                    nvec = len(xs)
                else:
                    nvec = xs.shape[0]
                out = [matvec(xs[i]) for i in range(nvec)]
                return out

            conv_k, evals_k, evecs_k = davidson_nosym1(gen_matvec, g, precond,
                             tol=self.conv_tol, max_cycle=self.max_cycle,
                             max_space=self.max_space, nroots=nroots, verbose=self.verbose)

            evals[k] = evals_k
            evecs[k] = evecs_k
            convs[k] = conv_k

            if nroots==1:
                evals_k = [evals_k]
                evecs_k = [evecs_k]
            for n in range(nroots):
                r1, r2 = self.vector_to_amplitudes_ea(evecs_k[n], kshift=kshift)
                if isinstance(r1, (tuple, list)):
                    qp_weight = sum([ri.norm()**2 for ri in r1])
                else:
                    qp_weight = r1.norm() **2
                logger.info(self, 'EOM-EA-CCSD root %d E = %.16g  qpwt = %0.6g',
                            n, evals_k[n], qp_weight)
        evecs = vstack(tuple(evecs)).reshape(len(kptlist),nroots,size)
        logger.timer(self, 'EOM-CCSD', *cput0)
        return evals, evecs

    def ipccsd_t_star(self, nroots=1, koopmans=True, guess=None, left=False,
                      eris=None, imds=None, partition=None, kptlist=None,
                      dtype=None, **kwargs):
        if not self.imds.made_t3p2_ip_imds:
            self.imds.make_t3p2_ip(self)
        return self.ipccsd(nroots, koopmans, guess, left,  eris, imds,
                           partition, kptlist, dtype, **kwargs)

    def eaccsd_t_star(self, nroots=1, koopmans=True, guess=None, left=False,
                      eris=None, imds=None, partition=None, kptlist=None,
                      dtype=None, **kwargs):
        if not self.imds.made_t3p2_ea_imds:
            self.imds.make_t3p2_ea(self)
        return self.eaccsd(nroots, koopmans, guess, left,  eris, imds,
                           partition, kptlist, dtype, **kwargs)

    def gen_sym(self, symbol, net=None):
        kpts = self.kpts
        gvec = self._scf.cell.reciprocal_vectors()
        sym = [symbol, [kpts]*(len(symbol)-symbol.count('0')), net, gvec]
        return sym

    def vector_to_amplitudes_ip(self, vector, kshift=0):
        kpti = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpti)
        sym2 = self.gen_sym('++-', kpti)
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = array(vector[:nocc], sym1)
        r2 = array(vector[nocc:].reshape(nkpts,nkpts,nocc,nocc,nvir), sym2)
        r1.symlib = r2.symlib = self.symlib
        return r1, r2

    def vector_to_amplitudes_ea(self, vector, kshift=0):
        kpti = self.kpts[kshift]
        sym1 = self.gen_sym('+', kpti)
        sym2 = self.gen_sym('-++', kpti)
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nmo - nocc
        r1 = array(vector[:nvir], sym1)
        r2 = array(vector[nvir:].reshape(nkpts,nkpts,nocc,nvir,nvir), sym2)
        r1.symlib = r2.symlib = self.symlib
        return r1, r2

    def ipccsd_diag(self, kshift):
        if not self.imds.made_ip_imds:
            self.imds.make_ip(self.ip_partition)
        imds = self.imds
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('++-', self.kpts[kshift])
        nkpts = self.nkpts
        nocc,nvir = imds.t1.shape
        t2 = imds.t2
        Hr1 = -imds.Loo.diagonal()[kshift]
        Hr1 = array(Hr1, sym1)

        IJB = self.symlib.get_irrep_map(sym2)
        all_tasks = [[ki,kj] for ki, kj in itertools.product(range(nkpts),repeat=2)]
        if self.ip_partition == 'mp':
            foo = imds.eris.foo.diagonal()
            fvv = imds.eris.fvv.diagonal()
            Hr2 = -foo.reshape(nkpts,1,nocc,1,1) - foo.reshape(1,nkpts,1,nocc) +\
                  einsum('Bb,IJB->IJb', fvv, IJB)
        else:
            lvv = imds.Lvv.diagonal()
            loo = imds.Loo.diagonal()
            wij = einsum('IJIijij->IJij', imds.Woooo)
            wjb = einsum('JBJjbjb,IJB->IJjb', imds.Wovov, IJB)
            wjb2 = einsum('JBBjbbj,IJB->IJjb', imds.Wovvo, IJB)
            wib = einsum('IBIibib,IJB->IJib', imds.Wovov, IJB)

            Hr2 = -loo.reshape(nkpts,1,nocc,1,1) - loo.reshape(1,nkpts,1,nocc,1) +\
                  einsum('Bb,IJB->IJb', lvv, IJB).reshape(nkpts,nkpts,1,1,nvir)

            Hr2+= (wij.reshape(nkpts,nkpts,nocc,nocc,1) - \
                   wjb.reshape(nkpts,nkpts,1,nocc,nvir) + \
                   2*wjb2.reshape(nkpts,nkpts,1,nocc,nvir) -\
                   wib.reshape(nkpts,nkpts,nocc,1,nvir))

            Hr2 = Hr2 - einsum('IJjb,IJ,ij->IJijb', wjb2, eye(nkpts), eye(nocc))
            Woovvtmp = imds.Woovv.transpose(0,1,3,2)[:,:,kshift]
            Hr2 -= 2.*einsum('IJijcb,JIjicb->IJijb', t2[:,:,kshift], Woovvtmp)
            Hr2 += einsum('IJijcb,IJijcb->IJijb', t2[:,:,kshift], Woovvtmp)
        Hr2 = array(Hr2, sym2)
        return self.amplitudes_to_vector_ip(Hr1, Hr2)

    def eaccsd_diag(self, kshift):
        if not self.imds.made_ea_imds:
            self.imds.make_ea(self.ea_partition)
        imds = self.imds
        sym1 = self.gen_sym('+', self.kpts[kshift])
        sym2 = self.gen_sym('-++', self.kpts[kshift])
        nkpts = self.nkpts
        nocc,nvir = imds.t1.shape
        t2 = imds.t2

        Hr1 = imds.Lvv.diagonal()[kshift]
        Hr1 = array(Hr1, sym1)

        JAB = self.symlib.get_irrep_map(sym2)

        if self.ea_partition == 'mp':
            foo = imds.eris.foo.diagonal()
            fvv = imds.eris.fvv.diagonal()
            Hr2 = -foo.reshape(nkpts,1,nocc,1,1) + fvv.reshape(1,nkpts,1,nvir,1) +\
                  einsum('Bb,JAB->JAb', fvv, JAB).reshape(nkpts,nkpts,1,1,nvir)
        else:
            loo = imds.Loo.diagonal()
            lvv = imds.Lvv.diagonal()
            wab = einsum("ABAabab,JAB->JAab", imds.Wvvvv, JAB)
            wjb = einsum('JBJjbjb,JAB->JAjb', imds.Wovov, JAB)
            wjb2 = einsum('JBBjbbj,JAB->JAjb', imds.Wovvo, JAB)
            wja = einsum('JAJjaja->JAja', imds.Wovov)

            Hr2 = -loo.reshape(nkpts,1,nocc,1,1) + lvv.reshape(1,nkpts,1,nvir,1) +\
                  einsum('Bb,JAB->JAb', lvv, JAB).reshape(nkpts,nkpts,1,1,nvir)

            Hr2 += (wab.reshape(nkpts,nkpts,1,nvir,nvir) - \
                    wjb.reshape(nkpts,nkpts,nocc,1,nvir) + \
                    2*wjb2.reshape(nkpts,nkpts,nocc,1,nvir) - \
                    wja.reshape(nkpts,nkpts,nocc,nvir,1))
            wjbtmp = einsum('JAAjbbj,JAA->JAjb', imds.Wovvo, JAB)
            Hr2 -= einsum('JAjb,ab->JAjab', wjbtmp, eye(nvir))
            Hr2 -= 2*einsum('JAijab,JAijab->JAjab', t2[kshift], imds.Woovv[kshift])
            Woovvtmp = imds.Woovv.transpose(0,1,3,2)[kshift]
            Hr2 += einsum('JAijab,JAijab->JAjab', t2[kshift], Woovvtmp)

        Hr2 = array(Hr2, sym2)
        return self.amplitudes_to_vector_ea(Hr1, Hr2)

    def ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift):
        from pyscf.ctfcc import kccsd_t_rhf
        return kccsd_t_rhf.ipccsd_star_contract(self, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, kshift)

    def eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift):
        from pyscf.ctfcc import kccsd_t_rhf
        return kccsd_t_rhf.eaccsd_star_contract(self, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, kshift)

    def ccsd_t(self, t1=None, t2=None, eris=None, slice_size=None):
        if slice_size is None: slice_size = self.slice_size
        from pyscf.ctfcc import kccsd_t_rhf
        return kccsd_t_rhf.kernel(self, eris, t1, t2, slice_size)

    def ccsd_t_slow(self, t1=None, t2=None, eris=None):
        raise NotImplementedError

def _get_eijab(ki, kj, ka, kconserv, script_mo_a, script_mo_b=None):
    if script_mo_b is None: script_mo_b = script_mo_a
    nocca, nvira, mo_e_oa, mo_e_va, nonzero_opaddinga, nonzero_vpaddinga = script_mo_a
    noccb, nvirb, mo_e_ob, mo_e_vb, nonzero_opaddingb, nonzero_vpaddingb = script_mo_b
    nkpts = kconserv.shape[0]
    kb = kconserv[ki,ka,kj]
    size = nocca*noccb*nvira*nvirb
    off = (ki*nkpts**2+kj*nkpts+ka)*size
    ind = off + np.arange(size)
    eia = kccsd_rhf._get_epq([0,nocca,ki,mo_e_oa,nonzero_opaddinga],
                   [0,nvira,ka,mo_e_va,nonzero_vpaddinga],
                   fac=[1.0,-1.0])
    ejb = kccsd_rhf._get_epq([0,noccb,kj,mo_e_ob,nonzero_opaddingb],
                   [0,nvirb,kb,mo_e_vb,nonzero_vpaddingb],
                   fac=[1.0,-1.0])
    val = eia[:,None,:,None] + ejb[None,:,None,:]
    return ind, val.ravel()

class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None):
        from pyscf.pbc import df
        from pyscf.pbc import tools
        from pyscf.pbc.cc.ccsd import _adjust_occ
        log = logger.Logger(cc.stdout, cc.verbose)
        cput0 = (time.clock(), time.time())
        self.cell = cell = cc._scf.cell
        self.kpts = kpts = cc.kpts
        nkpts = cc.nkpts
        nocc = cc.nocc
        nmo = cc.nmo
        nvir = nmo - nocc

        if mo_coeff is None:
            mo_coeff = cc.mo_coeff
        dtype = mo_coeff[0].dtype

        mo_coeff = self.mo_coeff = padded_mo_coeff(cc, mo_coeff)
        # Re-make our fock MO matrix elements from density and fock AO
        if rank==0:
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            exxdiv = cc._scf.exxdiv if cc.keep_exxdiv else None
            with lib.temporary_env(cc._scf, exxdiv=exxdiv):
            # _scf.exxdiv affects eris.fock. HF exchange correction should be
            # excluded from the Fock matrix.
                vhf = cc._scf.get_veff(cell, dm)
            fockao = cc._scf.get_hcore() + vhf
            self.fock = np.asarray([reduce(np.dot, (mo.T.conj(), fockao[k], mo))
                                for k, mo in enumerate(mo_coeff)])
            self.e_hf = cc._scf.energy_tot(dm=dm, vhf=vhf)

            self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]

            if not cc.keep_exxdiv:
                self.mo_energy = [self.fock[k].diagonal().real for k in range(nkpts)]
                madelung = tools.madelung(cell, kpts)
                self.mo_energy = [_adjust_occ(mo_e, nocc, -madelung)
                                  for k, mo_e in enumerate(self.mo_energy)]
        else:
            self.fock = self.e_hf = self.mo_energy = None
        ctf_helper.synchronize(self, ["fock", "e_hf", "mo_energy"])

        # Get location of padded elements in occupied and virtual space.
        nocc_per_kpt = cc.get_nocc(per_kpoint=True)
        nonzero_padding = padding_k_idx(cc, kind="joint")

        # Check direct and indirect gaps for possible issues with CCSD convergence.
        mo_e = [self.mo_energy[kp][nonzero_padding[kp]] for kp in range(nkpts)]
        mo_e = np.sort([y for x in mo_e for y in x])  # Sort de-nested array
        gap = mo_e[np.sum(nocc_per_kpt)] - mo_e[np.sum(nocc_per_kpt)-1]
        if gap < 1e-5:
            logger.warn(cc, 'HOMO-LUMO gap %s too small for KCCSD. '
                            'May cause issues in convergence.', gap)

        fao2mo = cc._scf.with_df.ao2mo
        kconserv = cc.khelper.kconserv
        khelper = cc.khelper
        self.dtype = dtype

        sym1= cc._sym[0]
        fock = asarray(self.fock)
        self.foo = array(fock[:,:nocc,:nocc], sym=sym1)
        self.fov = array(fock[:,:nocc,nocc:], sym=sym1)
        self.fvv = array(fock[:,nocc:,nocc:], sym=sym1)

        self.mo_e_o = mo_e_o = [e[:nocc] for e in self.mo_energy]
        self.mo_e_v = mo_e_v = [e[nocc:] + cc.level_shift for e in self.mo_energy]
        self._foo = asarray([np.diag(eo) for eo in mo_e_o])
        self._fvv = asarray([np.diag(ev) for ev in mo_e_v])
        self.eia = np.zeros([nkpts,nocc,nvir])
        nonzero_opadding, nonzero_vpadding = padding_k_idx(cc, kind="split")
        self.nonzero_opadding = nonzero_opadding
        self.nonzero_vpadding = nonzero_vpadding

        for ki in range(nkpts):
            self.eia[ki] = kccsd_rhf._get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                                              [0,nvir,ki,mo_e_v,nonzero_vpadding],
                                              fac=[1.0,-1.0])

        self.eia = asarray(self.eia)
        script_mo = (nocc, nvir, mo_e_o, mo_e_v, nonzero_opadding, nonzero_vpadding)
        get_eijab  = lambda ki,kj,ka: _get_eijab(ki, kj, ka, kconserv,script_mo)
        all_tasks = [[ki,kj,ka] for ki,kj,ka in itertools.product(range(nkpts), repeat=3)]
        self.eijab = frombatchfunc(get_eijab, (nocc,nocc,nvir,nvir), all_tasks, sym=cc._sym[1])

        if type(cc._scf.with_df) is df.FFTDF:
            ao2mo.make_fftdf_eris_rhf(cc, self)
        else:
            from pyscf.ctfcc.integrals import mpigdf
            if type(cc._scf.with_df) is mpigdf.GDF:
                ao2mo.make_df_eris_rhf(cc, self)
            elif type(cc._scf.with_df) is df.GDF:
                logger.warn(cc, "GDF converted to an MPIGDF object, \
                                   one process used for reading from disk")
                cc._scf.with_df = mpigdf.from_serial(cc._scf.with_df)
                ao2mo.make_df_eris_rhf(cc, self)
            else:
                raise NotImplementedError("DF object not recognized")
        logger.timer(cc, "ao2mo transformation", *cput0)

    def get_eijkab(self, kshift=0):
        nk, nocc, nvir = self.eia.shape
        all_tasks = [[ki,kj,kk,ka] for ki,kj,kk,ka in itertools.product(range(nk), repeat=4)]
        def _get_eijkab(ki,kj,kk,ka):
            kb = kpts_helper.get_kconserv3(self.cell, self.kpts, [ki,kj,kk,ka,kshift])
            eijk = _get_epqr([0,nocc,ki,self.mo_e_o,self.nonzero_opadding],
                             [0,nocc,kj,self.mo_e_o,self.nonzero_opadding],
                             [0,nocc,kk,self.mo_e_o,self.nonzero_opadding])

            eab = kccsd_rhf._get_epq([0,nvir,ka,mo_e_v,nonzero_vpadding],
                                     [0,nvir,kb,mo_e_v,nonzero_vpadding],
                                     fac=[1.0,1.0])

            eout = eijk[:,:,:,None,None] - eab[None,None,None]
            off = ki*nk**3+kj*nk**2+kk*nk+ka
            ind = off*eout.size+np.arange(eout.size)
            return ind, eout.ravel()
        shape = [nk,]*4+[nocc,]*3+[nvir,]*2
        eijkabc = frombatchfunc(_get_eijkab, shape, all_tasks).array
        return eijkabc

    def get_eijabc(self, kshift=0):
        nk, nocc, nvir = self.eia.shape
        all_tasks = [[ki,kj,ka,kb] for ki,kj,ka,kb in itertools.product(range(nk), repeat=4)]

        def _get_eijabc(ki,kj,ka,kb):
            kc = kpts_helper.get_kconserv3(self.cell, self.kpts, [ki,kj,kshift,ka,kb])
            eij = kccsd_rhf._get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                                     [0,nocc,kj,mo_e_o,nonzero_opadding],
                                     fac=[1.0,1.0])

            eabc = _get_epqr([0,nvir,ka,self.mo_e_v,self.nonzero_vpadding],
                             [0,nvir,kb,self.mo_e_v,self.nonzero_vpadding],
                             [0,nvir,kc,self.mo_e_v,self.nonzero_vpadding])

            eout = eij[:,:,None,None,None] - eabc[None,None]
            off = ki*nk**3+kj*nk**2+ka*nk+kb
            ind = off*eout.size+np.arange(eout.size)
            return ind, eout.ravel()

        shape = [nk,]*4+[nocc,]*2+[nvir,]*3
        eijkabc = frombatchfunc(_get_eijabc, shape, all_tasks).array
        return eijkabc

    def get_eijkabc(self):
        nk, nocc, nvir = self.eia.shape
        all_tasks = [[ki,kj,kk,ka,kb] for ki,kj,kk,ka,kb in itertools.product(range(nk), repeat=5)]
        def _get_eijkabc(ki,kj,kk,ka,kb):
            kc = kpts_helper.get_kconserv3(self.cell, self.kpts, [ki, kj, kk, ka, kb])
            eijk = _get_epqr([0,nocc,ki,self.mo_e_o,self.nonzero_opadding],
                             [0,nocc,kj,self.mo_e_o,self.nonzero_opadding],
                             [0,nocc,kk,self.mo_e_o,self.nonzero_opadding])

            eabc = _get_epqr([0,nvir,ka,self.mo_e_v,self.nonzero_vpadding],
                             [0,nvir,kb,self.mo_e_v,self.nonzero_vpadding],
                             [0,nvir,kc,self.mo_e_v,self.nonzero_vpadding])

            eout = eijk[:,:,:,None,None,None] - eabc[None,None,None]
            off = ki*nk**4+kj*nk**3+kk*nk**2+ka*nk+kb
            ind = off*eout.size+np.arange(eout.size)
            return ind, eout.ravel()
        shape = [nk,]*4+[nocc,]*3+[nvir,]*3
        eijkabc = frombatchfunc(_get_eijkabc, shape, all_tasks).array
        return eijkabc

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

    mycc = KRCCSD(kmf)
    mycc.kernel()

    eip, vip = mycc.ipccsd(nroots=2, kptlist=[1], left=True)
    eea, vea = mycc.eaccsd(nroots=2, kptlist=[2], left=True)

    print(eip[0,0] - 0.13448918)
    print(eip[0,1] - 0.48273254)

    print(eea[0,0] - 1.60938351)
    print(eea[0,1] - 2.22840054)
