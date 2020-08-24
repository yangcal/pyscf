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
from pyscf.ctfcc import kccsd_uhf
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

mycc = kccsd_uhf.KUCCSD(kmf)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)


def tearDownModule():
    global cell, kmf, eris, mycc
    cell.stdout.close()
    del cell, kmf, eris, mycc

class KnownValues(unittest.TestCase):

    def test_kuccsd(self):
        self.assertAlmostEqual(mycc.e_tot, -2.157799002010045, 7)

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
        self.assertAlmostEqual(eea[0,2], 1.754172351617824, 7)

    def test_ERIS(self):
        self.assertAlmostEqual(finger(eris.oooo), (-0.5663602029511005+0.13897557626183452j), 11)
        self.assertAlmostEqual(finger(eris.ooov), (0.008292871531888998-0.022535033331348906j),11)
        self.assertAlmostEqual(finger(eris.oovv), (-0.19745347320314238-0.1990131005612243j), 11)
        self.assertAlmostEqual(finger(eris.ovov), (-0.01787065927687745-0.009782903039961786j),11)
        self.assertAlmostEqual(finger(eris.voov), (0.2480170495503125-0.000341621912460309j), 11)
        self.assertAlmostEqual(finger(eris.vovv), (-0.015653841878366628+0.07808117864835849j),11)
        self.assertAlmostEqual(finger(eris.vvvv), (-0.28608507739589817-0.2789933410447869j), 11)

        self.assertAlmostEqual(finger(eris.ooOO), (-0.09322391615098397-0.012397081394196773j),11)
        self.assertAlmostEqual(finger(eris.ooOV), (-0.2616051311349463+0.04706972584581409j),11)
        self.assertAlmostEqual(finger(eris.ooVV), (0.03870663127685213+0.318852308013844j),11)
        self.assertAlmostEqual(finger(eris.ovOV), (-0.0004024978666814366-0.020049112652133737j),11)
        self.assertAlmostEqual(finger(eris.voOV), (-0.07267693652002004-0.01389474009658936j),11)
        self.assertAlmostEqual(finger(eris.voVV), (0.29524951566837754+0.1739391941115491j),11)
        self.assertAlmostEqual(finger(eris.vvVV), (0.16293566105419277+0.13574076271440355j),11)

        self.assertAlmostEqual(finger(eris.OOOO), (-0.13648959573292002+0.002854091330425734j), 11)
        self.assertAlmostEqual(finger(eris.OOOV), (-0.09817531006825315-0.01031471489069626j), 11)
        self.assertAlmostEqual(finger(eris.OOVV), (-0.4602901763741917+0.2542055427477413j), 11)
        self.assertAlmostEqual(finger(eris.OVOV), (0.07171766714712911-0.02244004897506647j), 11)
        self.assertAlmostEqual(finger(eris.VOOV), (-0.02990246690824215+0.04021282743046374j), 11)
        self.assertAlmostEqual(finger(eris.VOVV), (0.09408375386521832-0.27912958034354846j), 11)
        self.assertAlmostEqual(finger(eris.VVVV), (-0.5724148041822135-0.02322639856137669j), 11)

        self.assertAlmostEqual(finger(eris.OOov), (0.35689801667040644-0.03534764908633659j),11)
        self.assertAlmostEqual(finger(eris.OOvv), (-0.10281846193479374-0.1288747915675524j),11)
        self.assertAlmostEqual(finger(eris.VOov), (-0.043455917554529835+0.01604731239006j),11)
        self.assertAlmostEqual(finger(eris.VOvv), (-0.025762799468297324+0.010660582799866307j),11)

if __name__ == "__main__":
    print("Full Tests for KUCCSD")
    unittest.main()
