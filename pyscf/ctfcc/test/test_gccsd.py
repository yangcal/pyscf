#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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


import unittest
import copy
import numpy
from pyscf import gto, scf, lib
from pyscf.ctfcc import gccsd

def finger(array):
    return lib.finger(array.to_nparray())

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.verbose = 4
mol.output = '/dev/null'
mol.basis = '631g'
mol.spin = 2
mol.build()
mf = scf.UHF(mol).run(conv_tol=1e-12)
mf = scf.addons.convert_to_ghf(mf)

gcc1 = gccsd.GCCSD(mf)
gcc1.conv_tol = 1e-9
gcc1.kernel()


def tearDownModule():
    global mol, mf, gcc1
    mol.stdout.close()
    del mol, mf, gcc1

class KnownValues(unittest.TestCase):

    def test_gccsd(self):
        self.assertAlmostEqual(gcc1.e_corr, -0.10805861695870976, 7)

    def test_ERIS(self):
        gcc = gccsd.GCCSD(mf, frozen=4)
        numpy.random.seed(9)
        mo_coeff0 = numpy.random.random(mf.mo_coeff.shape) - .9
        nao = mo_coeff0.shape[0]//2
        orbspin = numpy.array([0,1,0,1,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,1,0,1,0,1])
        mo_coeff0[nao:,orbspin==0] = 0
        mo_coeff0[:nao,orbspin==1] = 0
        mo_coeff1 = mo_coeff0.copy()
        mo_coeff1[-1,0] = 1e-12

        eris = gcc.ao2mo(mo_coeff0)
        self.assertAlmostEqual(finger(eris.oooo),  15.97533838570434, 9)
        self.assertAlmostEqual(finger(eris.ooov), -80.97666019169982, 9)
        self.assertAlmostEqual(finger(eris.oovv), 278.00028168381675, 9)
        self.assertAlmostEqual(finger(eris.ovov),   2.34326750142844, 9)
        self.assertAlmostEqual(finger(eris.ovvv), 908.61659731634768, 9)
        self.assertAlmostEqual(finger(eris.vvvv), 756.77383112217694, 9)

        eris = gcc.ao2mo(mo_coeff1)
        self.assertAlmostEqual(finger(eris.oooo),  15.97533838570434, 9)
        self.assertAlmostEqual(finger(eris.ooov), -80.97666019169982, 9)
        self.assertAlmostEqual(finger(eris.oovv), 278.00028168381675, 9)
        self.assertAlmostEqual(finger(eris.ovov),   2.34326750142844, 9)
        self.assertAlmostEqual(finger(eris.ovvv), 908.61659731634768, 9)
        self.assertAlmostEqual(finger(eris.vvvv), 756.77383112217694, 9)

    def test_mbpt2(self):
        mygcc = gccsd.GCCSD(mf)
        e = mygcc.kernel(mbpt2=True)[0]
        self.assertAlmostEqual(e, -0.096257842171487293, 9)

    def test_ccsd_t(self):
        mygcc = gccsd.GCCSD(mf)
        mygcc.kernel()
        mygcc.max_memory = 4
        et1 = mygcc.ccsd_t()
        et2 = mygcc.ccsd_t_slow()
        self.assertAlmostEqual(et1, -0.001017602076596314, 9)
        self.assertAlmostEqual(et2, -0.001017602076596314, 9)

    def test_ipccsd(self):
        mycc = gccsd.GCCSD(mf)
        mycc.kernel()
        e, vrip = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.15082820837755678, 6)
        self.assertAlmostEqual(e[1], 0.19232763280983167, 6)
        self.assertAlmostEqual(e[2], 0.27128475534934055, 6)

        e, vlip = mycc.ipccsd(nroots=3, left=True)
        self.assertAlmostEqual(e[0], 0.15082820018186482, 6)
        self.assertAlmostEqual(e[1], 0.19232758023617233 , 6)
        self.assertAlmostEqual(e[2], 0.2712847467791727, 6)

        e = mycc.ipccsd_star_contract(e, vrip, vlip)
        self.assertAlmostEqual(e[0], 0.1518068055738176, 6)
        self.assertAlmostEqual(e[1], 0.15958467796748166 , 6)
        self.assertAlmostEqual(e[2], 0.23208658354685385, 6)

        e, vrip = mycc.ipccsd_t_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.15097704489378475, 6)
        self.assertAlmostEqual(e[1], 0.1925884002414439 , 6)
        self.assertAlmostEqual(e[2], 0.27165653727709854, 6)


    def test_eaccsd(self):
        mycc = gccsd.GCCSD(mf)
        mycc.kernel()

        e, vrea = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], -0.09881084664781134, 6)
        self.assertAlmostEqual(e[1], 0.012832111162019861, 6)
        self.assertAlmostEqual(e[2], 0.10735274803853091, 6)

        e, vlea = mycc.eaccsd(nroots=3, left=True)
        self.assertAlmostEqual(e[0], -0.09881086675283757, 6)
        self.assertAlmostEqual(e[1], 0.012832112558994009 , 6)
        self.assertAlmostEqual(e[2], 0.10735277332329218, 6)

        e = mycc.eaccsd_star_contract(e, vrea, vlea)
        self.assertAlmostEqual(e[0], -0.09621381070967173, 6)
        self.assertAlmostEqual(e[1], -0.08190303471855628 , 6)
        self.assertAlmostEqual(e[2], 0.013079651608785325, 6)

        e, vrea = mycc.eaccsd_t_star(nroots=3)
        self.assertAlmostEqual(e[0], -0.097406746441364, 6)
        self.assertAlmostEqual(e[1], 0.014114547427102279, 6)
        self.assertAlmostEqual(e[2], 0.10866286070714065, 6)

    def test_eeccsd(self):
        mycc=  gccsd.GCCSD(mf)
        mycc.kernel()
        e, v = mycc.eeccsd(nroots=4)
        self.assertAlmostEqual(e[0], -0.2875744347990707, 6)
        self.assertAlmostEqual(e[1], 7.093300636038563e-05, 6)
        self.assertAlmostEqual(e[2], 0.026861594481731924, 6)
        self.assertAlmostEqual(e[3], 0.045527827078651585, 6)

if __name__ == "__main__":
    print("Full Tests for GCCSD")
    unittest.main()
