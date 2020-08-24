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


import time
import numpy
import sys
import ctf
import os
import warnings
import scipy.linalg
from pyscf.lib import linalg_helper
from pyscf.lib import misc
from pyscf.lib import logger
from pyscf import __config__
from pyscf.ctfcc import ctf_helper

rank = ctf_helper.rank

DAVIDSON_LINDEP = getattr(__config__, 'lib_linalg_helper_davidson_lindep', 1e-14)
SORT_EIG_BY_SIMILARITY = \
        getattr(__config__, 'lib_linalg_helper_davidson_sort_eig_by_similiarity', False)
PROJECT_OUT_CONV_EIGS = \
        getattr(__config__, 'lib_linalg_helper_davidson_project_out_eigs', False)
FOLLOW_STATE = getattr(__config__, 'lib_linalg_helper_davidson_follow_state', False)


pick_real_eigs = linalg_helper.pick_real_eigs
_sort_by_similarity = linalg_helper._sort_by_similarity
_sort_elast = linalg_helper._sort_elast

def _len(x):
    if isinstance(x, (tuple,list)):
        nvec = len(x)
    else:
        nvec = x.shape[0]
    return nvec

def dot(a, b):
    out = ctf.dot(a, b)
    if out.size ==1:
        return out.to_nparray()
    else:
        return out

def eig(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
        lindep=DAVIDSON_LINDEP, callback=None,
        nroots=1, left=False, pick=pick_real_eigs,
        verbose=logger.WARN, follow_state=FOLLOW_STATE):

    def mult(xs):
        nvec = _len(xs)
        out = [aop(xs[i]) for i in range(nvec)]
        return out

    res = davidson_nosym1(mult,
                          x0, precond, tol, max_cycle, max_space, lindep,
                          callback, nroots, left, pick, verbose, follow_state)
    if left:
        e, vl, vr = res[1:]
        if nroots == 1:
            return e[0], vl[0], vr[0]
        else:
            return e, vl, vr
    else:
        e, x = res[1:]
        if nroots == 1:
            return e[0], x[0]
        else:
            return e, x

def davidson_nosym1(aop, x0, precond, tol=1e-12, max_cycle=50, max_space=12,
                    lindep=DAVIDSON_LINDEP, callback=None,
                    nroots=1, left=False, pick=pick_real_eigs,
                    verbose=logger.WARN, follow_state=FOLLOW_STATE,
                    tol_residual=None):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(sys.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if not callable(precond):
        raise ValueError("precond not initialized")

    if isinstance(x0, ctf.core.tensor) and x0.ndim == 1:
        x0 = [x0]

    max_space = max_space + (nroots-1) * 4

    log.debug1('max_cycle %d  max_space %d  incore True',
               max_cycle, max_space)
    dtype = None
    heff = None
    fresh_start = True
    e = 0
    v = None
    conv = [False] * nroots
    emin = None
    norm_min = 1
    for icyc in range(max_cycle):
        if fresh_start:
            xs = []
            ax = []
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            #xt = None
            x0len = _len(x0)
            xt, x0 = _qr(x0, lindep)[0], None
            if _len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.  The davidson may fail.'
                         'Check to see if `pick` function :%s: is providing linear dependent '
                         'vectors' % (x0len - _len(xt), pick.__name__))
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = [False] * nroots
        elif _len(xt) > 1:
            xt = _qr(xt, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most
        axt = aop(xt)
        for k in range(_len(xt)):
            xs.append(xt[k])
            ax.append(axt[k])

        rnow = _len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], xt[0])
            except IndexError:
                dtype = numpy.result_type(ax[0].dtype, xs[0].dtype)
        if heff is None:  # Lazy initilize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv
        for i in range(rnow):
            for k in range(rnow):
                heff[head+k,head+i] = dot(xt[k].conj(), axt[i])

        for i in range(head):
            for k in range(rnow):
                heff[head+k,i] = dot(xt[k].conj(), ax[i])
                heff[i,head+k] = dot(xs[i].conj(), axt[k])


        w, v = scipy.linalg.eig(heff[:space,:space])
        w, v, idx = pick(w, v, nroots, locals())
        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin,
                                       heff[:space,:space])
            if e.size != elast.size:
                de = e
            else:
                de = e - elast
        else:
            e = w[:nroots]
            v = v[:,:nroots]

        x0 = _gen_x0(v, xs)
        ax0 = _gen_x0(v, ax)


        if SORT_EIG_BY_SIMILARITY:
            dx_norm = [0] * nroots
            xt = [None] * nroots
            for k, ek in enumerate(e):
                if not conv[k]:
                    xt[k] = ax0[k] - ek * x0[k]
                    dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                    if abs(de[k]) < tol and dx_norm[k] < toloose:
                        log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                                  k, dx_norm[k], ek, de[k])
                        conv[k] = True
        else:
            elast, conv_last = _sort_elast(elast, conv_last, vlast, v,
                                           fresh_start, log)
            de = e - elast
            dx_norm = []
            xt = []
            for k, ek in enumerate(e):
                xt.append(ax0[k] - ek * x0[k])
                dx_norm.append(numpy.sqrt(dot(xt[k].conj(), xt[k]).real))
                if not conv_last[k] and abs(de[k]) < tol and dx_norm[k] < toloose:
                    log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                              k, dx_norm[k], ek, de[k])
            dx_norm = numpy.asarray(dx_norm)
            conv = (abs(de) < tol) & (dx_norm < toloose)
        ax0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots*3):

            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide], norm_min)
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        if any(((not conv[k]) and n**2>lindep) for k, n in enumerate(dx_norm)):
            for k, ek in enumerate(e):
                if (not conv[k]) and dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
                    log.debug1('Throwing out eigenvector %d with norm=%4.3g', k, dx_norm[k])
        else:
            for k, ek in enumerate(e):
                if dx_norm[k]**2 > lindep:
                    xt[k] = precond(xt[k], e[0], x0[k])
                    xt[k] *= 1/numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
                else:
                    xt[k] = None
        xt = [xi for xi in xt if xi is not None]
        for i in range(space):
            for xi in xt:
                xi -= xs[i] * dot(xs[i].conj(), xi)

        norm_min = 1
        for i,xi in enumerate(xt):
            norm = numpy.sqrt(dot(xi.conj(), xi).real)
            if norm**2 > lindep:
                xt[i] *= 1/norm
                norm_min = min(norm_min, norm)
            else:
                xt[i] = None
        xt = [xi for xi in xt if xi is not None]
        xi = None
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if _len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                     dx_norm)
            conv = [conv[k] or (norm < toloose) for k,norm in enumerate(dx_norm)]
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    xnorm = numpy.array([ctf.norm(x0[i]) for i in range(_len(x0))])
    enorm = xnorm < 1e-6
    if numpy.any(enorm):
        if rank==0:
            warnings.warn("{:d} davidson root{_s}: {} {_has} very small norm{_s}: {}".format(
            enorm.sum(),
            ", ".join("#{:d}".format(i) for i in numpy.argwhere(enorm)[:, 0]),
            ", ".join("{:.3e}".format(i) for i in xnorm[enorm]),
            _s='s' if enorm.sum() > 1 else "",
            _has="have" if enorm.sum() > 1 else "has a",
        ))

    if left:
        if rank==0:
            warnings.warn('Left eigenvectors from subspace diagonalization method may not be converged')
        w, vl, v = scipy.linalg.eig(heff[:space,:space], left=True)
        e, v, idx = pick(w, v, nroots, locals())
        xl = _gen_x0(vl[:,idx[:nroots]].conj(), xs)
        x0 = _gen_x0(v[:,:nroots], xs)
        return numpy.asarray(conv), e[:nroots], xl, x0
    else:
        return numpy.asarray(conv), e, x0

def _qr(xs, lindep=1e-14):
    if isinstance(xs, (tuple, list)):
        nvec = len(xs)
    else:
        nvec = xs.shape[0]
    dtype = xs[0].dtype
    qs = [None]*nvec
    rmat = numpy.empty((nvec,nvec), order='F', dtype=dtype)
    nv = 0
    for i in range(nvec):
        xi = xs[i].copy()
        rmat[:,nv] = 0
        rmat[nv,nv] = 1
        for j in range(nv):
            prod = dot(qs[j].conj(),xi)
            xi -= qs[j]*prod
            rmat[:,nv] -= rmat[:,j] *prod
        innerprod = dot(xi.conj(),xi)
        norm = numpy.sqrt(innerprod)
        if innerprod > lindep:
            qs[nv] = xi/norm
            rmat[:nv+1,nv] /= norm
            nv += 1
    return qs[:nv], numpy.linalg.inv(rmat[:nv,:nv])


def _gen_x0(v, xs):
    space, nroots = v.shape
    vtmp = ctf.astensor(v)
    x0 = ctf.einsum('c,x->cx', vtmp[space-1], xs[space-1])
    for i in range(space-1):
        x0 += ctf.einsum('k,v->kv', vtmp[i], xs[i])
    return x0



def eigs(matvec, vecsize, nroots, x0=None, Adiag=None, guess=False, verbose=4):
    '''Davidson diagonalization method to solve A c = E c
    when A is not Hermitian.
    '''
    def matvec_args(vec, args=None):
        return matvec(vec)
    nroots = min(nroots, vecsize)
    conv, e, c = davidson(matvec, vecsize, nroots, x0, Adiag, verbose)
    return conv, e, c


def davidson(mult_by_A, N, neig, x0=None, Adiag=None, verbose=4, **kwargs):
    """Diagonalize a matrix via non-symmetric Davidson algorithm.

    mult_by_A() is a function which takes a vector of length N
        and returns a vector of length N.
    neig is the number of eigenvalues requested
    """
    log = logger.Logger(sys.stdout, verbose)
    from symtensor.ctf.backend import argsort
    cput1 = (time.clock(), time.time())
    Mmin = min(neig,N)
    Mmax = min(2*N,2000)
    tol = kwargs.get('conv_tol', 1e-6)

    def mult(arg):
        return mult_by_A(arg)

    if Adiag is None:
        raise ValueError("Adiag is not provided")

    idx = argsort(Adiag.real(), Mmin)
    lamda_k_old = 0
    lamda_k = 0
    target = 0
    conv = False
    if x0 is not None:
        assert (x0.shape == (Mmin, N) )
        b = x0.copy()
        Ab = tuple([mult(b[m]) for m in range(Mmin)])
        Ab = ctf.vstack(Ab).transpose()
    evals = numpy.zeros(neig,dtype=numpy.complex)
    evecs = []

    for istep,M in enumerate(range(Mmin,Mmax+1)):
        if M == Mmin:
            b = ctf.zeros((N,M))
            if rank==0:
                ind = [i*M+m for m,i in zip(range(M),idx)]
                fill = numpy.ones(len(ind))
                b.write(ind, fill)
            else:
                b.write([],[])
            Ab = tuple([mult(b[:,m]) for m in range(M)])
            Ab = ctf.vstack(Ab).transpose()
        else:
            Ab = ctf.hstack((Ab, mult(b[:,M-1]).reshape(N,-1)))
        Atilde = ctf.dot(b.conj().transpose(),Ab)
        Atilde = Atilde.to_nparray()

        lamda, alpha = diagonalize_asymm(Atilde)
        lamda_k_old, lamda_k = lamda_k, lamda[target]
        alpha_k = ctf.astensor(alpha[:,target])
        if M == Mmax:
            break
        q = ctf.dot( Ab-lamda_k*b, alpha_k)
        qnorm = ctf.norm(q)
        log.info('davidson istep = %d  root = %d  E = %.15g  dE = %.9g  residual = %.6g',
                 istep, target, lamda_k.real, (lamda_k - lamda_k_old).real, qnorm)
        cput1 = log.timer('davidson iter', *cput1)

        if ctf.norm(q) < tol:
            evecs.append(ctf.dot(b,alpha_k))
            evals[target] = lamda_k
            if target == neig-1:
                conv = True
                break
            else:
                target += 1
        eps = 1e-10
        xi = q/(lamda_k-Adiag+eps)
        bxi,R = ctf.qr(ctf.hstack((b,xi.reshape(N,-1))))
        nlast = bxi.shape[-1] - 1
        b = ctf.hstack((b,bxi[:,nlast].reshape(N,-1))) #can not replace nlast with -1, (inconsistent between numpy and ctf)
    evecs = ctf.vstack(tuple(evecs))
    return conv, evals.real, evecs

def diagonalize_asymm(H):
    E,C = numpy.linalg.eig(H)
    idx = E.real.argsort()
    E = E[idx]
    C = C[:,idx]
    return E,C

del DAVIDSON_LINDEP

if __name__ == '__main__':

    numpy.random.seed(12)
    n = 500
    a = numpy.arange(n*n).reshape(n,n)
    a = numpy.sin(numpy.sin(a))
    a = a + a.T + numpy.diag(numpy.random.random(n))*10
    b = numpy.random.random((n,n))
    b = numpy.dot(b,b.T) + numpy.eye(n)*5

    x0 = [a[0]/numpy.linalg.norm(a[0]),
          a[1]/numpy.linalg.norm(a[1]),]
    abdiag = numpy.dot(a,b).diagonal().copy()

    a = ctf.astensor(a)
    b = ctf.astensor(b)
    abdiag = ctf.astensor(abdiag)
    x0 = [ctf.astensor(i) for i in x0]
    x0 = ctf.vstack((tuple(x0)))

    def abop(x):
        return dot(a, dot(b, x))

    def precond(r, e0, x0):
        return r / (abdiag-e0)
    e0, x0 = eig(abop, x0, precond, max_cycle=100, max_space=30, verbose=5,
                 nroots=4, pick=pick_real_eigs)
    print(e0[0] - -10994.102910245942)
    print(e0[1] - -8420.800733331896)
    print(e0[2] - -190.184633050791)
    print(e0[3] - -156.187975856133)

    e0, vl, vr = eig(abop, x0, precond, max_cycle=100, max_space=30, verbose=5,
                     nroots=4, pick=pick_real_eigs, left=True)
    print(e0[0] - -10994.102910245976)
    print(e0[1] - -8420.800733331907)
    print(e0[2] - -190.18463305079075)
    print(e0[3] - -156.1879758561344)

    print(abs(vr[0].to_nparray()).sum() -18.11065949832812)
    print(abs(vr[1].to_nparray()).sum() -18.195952397318976)
    print(abs(vr[2].to_nparray()).sum() -17.797642389366803)
    print(abs(vr[3].to_nparray()).sum() -17.77919657644214)
