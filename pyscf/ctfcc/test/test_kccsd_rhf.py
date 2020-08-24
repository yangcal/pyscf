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
from pyscf.pbc import gto, scf
from pyscf.ctfcc import kccsd_rhf
from pyscf import lib

def finger(array):
    if isinstance(array, numpy.ndarray):
        return lib.finger(array)
    elif hasattr(array, 'array'):
        return lib.finger(array.array.to_nparray())
    else:
        return lib.finger(array.to_nparray())

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
cell.output = '/dev/null'
cell.unit = 'B'
cell.verbose = 4
cell.build()

numpy.random.seed(1)
kpts = cell.make_kpts([1,1,2]) + numpy.random.random([3]) * 0.1
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
kmf.conv_tol = 1e-10
kmf.kernel()

mycc = kccsd_rhf.KRCCSD(kmf)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)

def tearDownModule():
    global cell, kmf, eris, mycc
    cell.stdout.close()
    del cell, kmf, eris, mycc

class KnownValues(unittest.TestCase):

    def test_krccsd(self):
        self.assertAlmostEqual(mycc.e_tot, -3.89002158177598, 7)

    def test_ERIS(self):
        self.assertAlmostEqual(finger(eris.oooo), -0.5186613775080537+0.08542815763593005j, 11)
        self.assertAlmostEqual(finger(eris.ooov), -0.14206075615913608+0.019762775097675548j, 11)
        self.assertAlmostEqual(finger(eris.ovov), -0.25322649922338636+0.016098684735699028j, 11)
        self.assertAlmostEqual(finger(eris.oovv), -0.02435531151401958+0.06840536481183612j, 11)
        self.assertAlmostEqual(finger(eris.ovvo), -0.03872301828409198-0.02361759314620874j, 11)
        self.assertAlmostEqual(finger(eris.ovvv), -0.0035795101052934902+0.005624995754349988j, 11)
        self.assertAlmostEqual(finger(eris.vvvv), -0.039821837954347024+0.1693360210426124j, 11)

    def test_ccsd_t(self):
        et = mycc.ccsd_t(eris=eris)
        self.assertAlmostEqual(et, -5.763819082524832e-06, 7)

    def test_ipccsd(self):
        mycc = kccsd_rhf.KRCCSD(kmf)
        mycc.kernel(eris=eris)

        eip, vipr = mycc.ipccsd(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eip[0,0], -0.012845248449557874, 6)
        self.assertAlmostEqual(eip[0,1], 0.24869281798951465, 6)

        eip, vipl = mycc.ipccsd(nroots=2, left=True, kptlist=[1])
        self.assertAlmostEqual(eip[0,0], -0.0128452483682340824, 6)
        self.assertAlmostEqual(eip[0,1], 0.24869299020699145, 6)

        eip = mycc.ipccsd_star_contract(eip[0], vipr[0], vipl[0], kshift=1)
        self.assertAlmostEqual(eip[0], -0.013161564620606655, 6)
        self.assertAlmostEqual(eip[1], 0.24840392419034982, 6)

        eip, vipr = mycc.ipccsd_t_star(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eip[0,0], -0.012842649996364289, 6)
        self.assertAlmostEqual(eip[0,1], 0.24869316458580962, 6)

    def test_eaccsd(self):
        mycc = kccsd_rhf.KRCCSD(kmf)
        mycc.kernel(eris=eris)
        eea, vear = mycc.eaccsd(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eea[0,0], 1.6919707688893464, 7)
        self.assertAlmostEqual(eea[0,1], 2.0227928652560507, 7)

        eea, veal = mycc.eaccsd(nroots=2, left=True, kptlist=[1])
        self.assertAlmostEqual(eea[0,0], 1.691970693260425, 7)
        self.assertAlmostEqual(eea[0,1], 2.0227928758095888, 7)

        eea = mycc.eaccsd_star_contract(eea[0], vear[0], veal[0], kshift=1)
        self.assertAlmostEqual(eea[0], 1.6919952046957227, 7)
        self.assertAlmostEqual(eea[1], 2.0226572829376566, 7)

        eea, vear = mycc.eaccsd_t_star(nroots=2, kptlist=[1])
        self.assertAlmostEqual(eea[0,0], 1.6919744277713589, 7)
        self.assertAlmostEqual(eea[0,1], 2.022795891301815, 7)


if __name__ == "__main__":
    print("Full Tests for KRCCSD")
    unittest.main()
