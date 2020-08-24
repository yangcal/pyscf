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
from pyscf import gto, scf, lib, cc
from pyscf.cc import uccsd_slow

finger = lib.finger

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '631g'
mol.spin = 2
mol.verbose = 5
mol.output = '/dev/null'
mol.build()
mf = scf.UHF(mol).run()
mycc = uccsd_slow.UCCSD(mf)
eris = mycc.ao2mo()
mycc.kernel(eris=eris)


def tearDownModule():
    global mol, mf, eris, mycc
    mol.stdout.close()
    mol.stdout.close()
    del mol, mf, eris, mycc

class KnownValues(unittest.TestCase):

    def test_ERIS(self):
        ucc1 = uccsd_slow.UCCSD(mf)
        nao,nmo = mf.mo_coeff[0].shape
        numpy.random.seed(1)
        mo_coeff = numpy.random.random((2,nao,nmo))
        eris = ucc1.ao2mo(mo_coeff)

        self.assertAlmostEqual(finger(eris.oooo), 44.60374162913075,11)
        self.assertAlmostEqual(finger(eris.ooov),-69.2362396219225, 11)
        self.assertAlmostEqual(finger(eris.ovov),-44.03062513701185, 11)
        self.assertAlmostEqual(finger(eris.oovv),-53.07194017842969, 11)
        self.assertAlmostEqual(finger(eris.voov),-88.33210110417593, 11)
        self.assertAlmostEqual(finger(eris.vovv),24.159087118678276, 11)
        self.assertAlmostEqual(finger(eris.vvvv),13.118949707895478, 11)
        self.assertAlmostEqual(finger(eris.OOOO),-52.539617168071416, 11)
        self.assertAlmostEqual(finger(eris.OOOV),-87.00143069691399, 11)
        self.assertAlmostEqual(finger(eris.OVOV),99.0206266475395, 11)
        self.assertAlmostEqual(finger(eris.OOVV),117.00928812610958, 11)
        self.assertAlmostEqual(finger(eris.VOOV),148.23572092177832, 11)
        self.assertAlmostEqual(finger(eris.VOVV),141.60334328876723, 11)
        self.assertAlmostEqual(finger(eris.VVVV),-164.7359622396963, 11)
        self.assertAlmostEqual(finger(eris.ooOO),0.27322612086597076, 11)
        self.assertAlmostEqual(finger(eris.OOov),-67.32829910380818, 11)
        self.assertAlmostEqual(finger(eris.ovOV),153.11904193962417, 11)
        self.assertAlmostEqual(finger(eris.ooVV),14.759150908730454, 11)
        self.assertAlmostEqual(finger(eris.vvVV),-188.78198969394785, 11)
        self.assertAlmostEqual(finger(eris.VOov),-114.48109530885833, 11)
        self.assertAlmostEqual(finger(eris.voVV),76.56909649778656, 11)
        self.assertAlmostEqual(finger(eris.ooOV),32.71966171741167, 11)
        self.assertAlmostEqual(finger(eris.OOvv),-40.62800685016493, 11)
        self.assertAlmostEqual(finger(eris.voOV),23.060535888168616, 11)
        self.assertAlmostEqual(finger(eris.VOvv),-31.61752563452297, 11)

    def test_uccsd_frozen(self):
        ucc1 = copy.copy(mycc)
        ucc1.frozen = 1
        self.assertEqual(ucc1.nmo, (12,12))
        self.assertEqual(ucc1.nocc, (5,3))
        ucc1.frozen = [0,1]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (4,2))
        ucc1.frozen = [[0,1], [0,1]]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (4,2))
        ucc1.frozen = [1,9]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (5,3))
        ucc1.frozen = [[1,9], [1,9]]
        self.assertEqual(ucc1.nmo, (11,11))
        self.assertEqual(ucc1.nocc, (5,3))
        ucc1.frozen = [9,10,12]
        self.assertEqual(ucc1.nmo, (10,10))
        self.assertEqual(ucc1.nocc, (6,4))
        ucc1.nmo = (13,12)
        ucc1.nocc = (5,4)
        self.assertEqual(ucc1.nmo, (13,12))
        self.assertEqual(ucc1.nocc, (5,4))

    def test_uccsd_frozen(self):
        # Freeze 1s electrons
        frozen = [[0,1], [0,1]]
        ucc = uccsd_slow.UCCSD(mf, frozen=frozen)
        ucc.diis_start_cycle = 1
        ecc, t1, t2 = ucc.kernel()
        self.assertAlmostEqual(ecc, -0.07414978284611283, 8)

    def test_vector_to_amplitudes(self):
        t1, t2 = mycc.vector_to_amplitudes(mycc.amplitudes_to_vector(mycc.t1, mycc.t2))
        self.assertAlmostEqual(abs(t1[0]-mycc.t1[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t1[1]-mycc.t1[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[0]-mycc.t2[0]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[1]-mycc.t2[1]).max(), 0, 12)
        self.assertAlmostEqual(abs(t2[2]-mycc.t2[2]).max(), 0, 12)

    def test_mbpt2(self):
        myucc = uccsd_slow.UCCSD(mf)
        e = myucc.kernel(mbpt2=True)[0]
        self.assertAlmostEqual(e, -0.096257842171487, 10)

    def test_ccsd_t(self):
        mycc = uccsd_slow.UCCSD(mf)
        mycc.kernel()
        et= mycc.ccsd_t_slow()
        self.assertAlmostEqual(et, -0.0010176027505768013, 8)

    def test_ipccsd(self):
        mycc = uccsd_slow.UCCSD(mf)
        mycc.kernel()
        e, v = mycc.ipccsd(nroots=4)
        self.assertAlmostEqual(e[0], 0.15082821014476724, 8)
        self.assertAlmostEqual(e[1], 0.27128474878779196, 8)
        self.assertAlmostEqual(e[2], 0.4594230554796196, 8)
        self.assertAlmostEqual(e[3], 0.6233541619027493, 8)


    def test_eaccsd(self):
        mycc = uccsd_slow.UCCSD(mf)
        mycc.kernel()
        e, v = mycc.eaccsd(nroots=4)
        self.assertAlmostEqual(e[0], -0.09881085219916441, 8)
        self.assertAlmostEqual(e[1], 0.094440274855227, 8)
        self.assertAlmostEqual(e[2], 0.1739431181202621, 8)
        self.assertAlmostEqual(e[3], 0.1956993723111896, 8)

if __name__ == "__main__":
    print("Full Tests for UCCSD")
    unittest.main()
