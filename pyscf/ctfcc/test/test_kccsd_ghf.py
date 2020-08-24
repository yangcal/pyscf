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
from pyscf.pbc import gto, scf, cc
from pyscf.ctfcc import kccsd
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
H 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.output = '/dev/null'
cell.unit = 'B'
cell.spin = 2
cell.verbose = 4
cell.build()

numpy.random.seed(1)
kpts = cell.make_kpts([1,1,2]) + numpy.random.random([3]) * 0.1
kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None)
kmf.conv_tol = 1e-10
kmf.kernel()

kmf = kmf.to_ghf(kmf)

mycc = kccsd.KGCCSD(kmf)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)

def tearDownModule():
    global cell, kmf, eris, mycc
    cell.stdout.close()
    del cell, kmf, eris, mycc

class KnownValues(unittest.TestCase):

    def test_kgccsd(self):
        self.assertAlmostEqual(mycc.e_tot, -2.157799002681663, 7)

    def test_ccsd_t(self):
        et = mycc.ccsd_t(eris=eris)
        self.assertAlmostEqual(et, -8.313908415835933e-06, 7)

    def test_ipccsd(self):
        eip, vip = mycc.ipccsd(nroots=3, kptlist=[1])
        self.assertAlmostEqual(eip[0,0], -0.2633831483774091, 7)
        self.assertAlmostEqual(eip[0,1], 0.15150435450695132, 7)
        self.assertAlmostEqual(eip[0,2], 0.159834792494711, 7)

    def test_eaccsd(self):
        eea, vea = mycc.eaccsd(nroots=3, kptlist=[1])
        self.assertAlmostEqual(eea[0,0], 0.5421663393754494, 7)
        self.assertAlmostEqual(eea[0,1], 1.402213688796914, 7)
        self.assertAlmostEqual(eea[0,2], 1.7563812281201503, 7)

    def test_ERIS(self):
        self.assertAlmostEqual(finger(eris.oooo), (0.1519560044428263+0.07078769444216403j), 11)
        self.assertAlmostEqual(finger(eris.ooov), (0.19506907485917468+0.01901317447802378j), 11)
        self.assertAlmostEqual(finger(eris.oovv), (-0.29946406270945974+0.015554839803996587j), 11)
        self.assertAlmostEqual(finger(eris.ovov), (0.06853196795051875-0.10201697827030454j), 11)
        self.assertAlmostEqual(finger(eris.ovvo), -1.429668268419472-0.14400838385199216j, 11)
        self.assertAlmostEqual(finger(eris.ovvv), (-0.006997024945202532+0.9004212747717855j), 11)
        self.assertAlmostEqual(finger(eris.vvvv), (-0.23217102222028152-2.0169831946010497j), 11)

if __name__ == "__main__":
    print("Full Tests for KGCCSD")
    unittest.main()
