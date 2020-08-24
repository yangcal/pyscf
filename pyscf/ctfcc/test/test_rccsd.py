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
from pyscf.ctfcc import rccsd
from pyscf.cc import rccsd_slow
import ctf
asarray = ctf.astensor

def finger(array):
    return lib.finger(array.to_nparray())

mol = gto.Mole()
mol.verbose = 4
mol.output = '/dev/null'
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]

mol.basis = '631g'
mol.build()
mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-8
mf.kernel()

mycc = rccsd.RCCSD(mf)
mycc.conv_tol = 1e-10
eris = mycc.ao2mo()
mycc.kernel(eris=eris)

def tearDownModule():
    global mol, mf, eris, mycc
    mol.stdout.close()
    del mol, mf, eris, mycc


class KnownValues(unittest.TestCase):

    def test_rccsd(self):
        mf = scf.RHF(mol).run()

        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        self.assertAlmostEqual(mycc.e_tot, -76.119346385357446, 7)

    def test_ERIS(self):
        mycc = rccsd.RCCSD(mf)
        numpy.random.seed(1)
        mo_coeff = numpy.random.random(mf.mo_coeff.shape)
        eris = mycc.ao2mo(mo_coeff)
        self.assertAlmostEqual(finger(eris.oooo),  4.963884938282539, 11)
        self.assertAlmostEqual(finger(eris.ooov), 21.353621010332812, 11)
        self.assertAlmostEqual(finger(eris.ovov),125.815506844421580, 11)
        self.assertAlmostEqual(finger(eris.oovv), 55.123681017639463, 11)
        self.assertAlmostEqual(finger(eris.ovvo),133.480835278982620, 11)
        self.assertAlmostEqual(finger(eris.ovvv), 95.756230114113222, 11)
        self.assertAlmostEqual(finger(eris.vvvv),-10.450387490987071, 11)

    def test_ccsd_t(self):
        mycc.max_memory = 0
        e = mycc.ccsd_t()
        self.assertAlmostEqual(e, -0.0009964234049929792, 10)

    def test_ccsd_t_slow(self):
        e = mycc.ccsd_t_slow()
        self.assertAlmostEqual(e, -0.0009964234049929792, 10)

    def test_mbpt2(self):
        e = mycc.kernel(mbpt2=True)[0]
        self.assertAlmostEqual(e, -0.12886859466216125, 10)

    def test_no_diis(self):
        cc1 = rccsd.RCCSD(mf)
        cc1.diis = False
        cc1.max_cycle = 4
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13516622806104395, 8)

    def test_iterative_dampling(self):
        cc1 = rccsd.RCCSD(mf)
        cc1.max_cycle = 3
        cc1.iterative_damping = 0.7
        cc1.kernel()
        self.assertAlmostEqual(cc1.e_corr, -0.13508743605375528, 8)

    def test_amplitudes_to_vector(self):
        vec = mycc.amplitudes_to_vector(mycc.t1, mycc.t2)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        self.assertAlmostEqual(abs((r1-mycc.t1).to_nparray()).max(), 0, 14)
        self.assertAlmostEqual(abs((r2-mycc.t2).to_nparray()).max(), 0, 14)

        vec = numpy.random.random(vec.size)
        vec = asarray(vec)
        r1, r2 = mycc.vector_to_amplitudes(vec)
        vec1 = mycc.amplitudes_to_vector(r1, r2)
        self.assertAlmostEqual(abs(vec-vec1).to_nparray().max(), 0, 14)

    def test_rccsd_frozen(self):
        cc1 = copy.copy(mycc)
        cc1.frozen = 1
        self.assertEqual(cc1.nmo, 12)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [0,1]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 3)
        cc1.frozen = [1,9]
        self.assertEqual(cc1.nmo, 11)
        self.assertEqual(cc1.nocc, 4)
        cc1.frozen = [9,10,12]
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 5)
        cc1.nmo = 10
        cc1.nocc = 6
        self.assertEqual(cc1.nmo, 10)
        self.assertEqual(cc1.nocc, 6)

    def test_ipccsd(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        e,v = mycc.ipccsd(nroots=1, left=False, koopmans=False)
        self.assertAlmostEqual(e, 0.42789082283297164, 6)

        e,v = mycc.ipccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.42789082283297275, 6)
        self.assertAlmostEqual(e[1], 0.5022685871351666, 6)
        self.assertAlmostEqual(e[2], 0.6855064009375147, 6)

        lv = mycc.ipccsd(nroots=3, left=True)[1]
        e = mycc.ipccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.43586153567672187, 6)
        self.assertAlmostEqual(e[1], 0.5095768096329091, 6)
        self.assertAlmostEqual(e[2], 0.6901486610248043, 6)

    def test_ipccsd_koopmans(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        e,v = mycc.ipccsd(nroots=3, koopmans=True)

        self.assertAlmostEqual(e[0], 0.42789082283297275, 6)
        self.assertAlmostEqual(e[1], 0.5022685871351666, 6)
        self.assertAlmostEqual(e[2], 0.6855064009375147, 6)

        e,v = mycc.ipccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.42789082574230336, 6)
        self.assertAlmostEqual(e[1], 0.5022685985481095, 6)
        self.assertAlmostEqual(e[2], 0.6855064073105679, 6)

    def test_ipccsd_partition(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()

        e,v = mycc.ipccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.4139194826856656, 6)
        self.assertAlmostEqual(e[1], 0.49042593403399176, 6)
        self.assertAlmostEqual(e[2], 0.674646596367033, 6)

        e,v = mycc.ipccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.42181955035571717, 6)
        self.assertAlmostEqual(e[1], 0.4964592112539948, 6)
        self.assertAlmostEqual(e[2], 0.6807327427036325, 6)

        e,v = mycc.ipccsd(nroots=3, partition='full', left=True)
        self.assertAlmostEqual(e[0], 0.41391948212580987, 6)
        self.assertAlmostEqual(e[1], 0.4904259343761573, 6)
        self.assertAlmostEqual(e[2], 0.6746465902577271, 6)

        e,v = mycc.ipccsd(nroots=3, partition='mp', left=True)
        self.assertAlmostEqual(e[0], 0.4218195503557185, 6)
        self.assertAlmostEqual(e[1], 0.4964592112329121, 6)
        self.assertAlmostEqual(e[2], 0.6807327919181598, 6)

    def test_ipccsd_Ta(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        e, v = mycc.ipccsd_t_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.42812336038230514, 6)
        self.assertAlmostEqual(e[1], 0.502532883518334, 6)
        self.assertAlmostEqual(e[2], 0.6856410648186314 , 6)
        e, v = mycc.ipccsd_t_star(nroots=3, left=True)
        self.assertAlmostEqual(e[0], 0.42812335090016695, 6)
        self.assertAlmostEqual(e[1], 0.5025329319118513, 6)
        self.assertAlmostEqual(e[2], 0.6856410726317309 , 6)

    def test_eaccsd(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        e,v = mycc.eaccsd(nroots=1, left=False, koopmans=False)
        self.assertAlmostEqual(e, 0.19050587834891175, 6)

        e,v = mycc.eaccsd(nroots=3)
        self.assertAlmostEqual(e[0], 0.19050587834891125, 6)
        self.assertAlmostEqual(e[1], 0.2834522317278694, 6)
        self.assertAlmostEqual(e[2], 0.5228067544581533, 6)

        lv = mycc.eaccsd(nroots=3, left=True)[1]
        e = mycc.eaccsd_star_contract(e, v, lv)
        self.assertAlmostEqual(e[0], 0.1894169012932152, 6)
        self.assertAlmostEqual(e[1], 0.2820757006580406, 6)
        self.assertAlmostEqual(e[2], 0.4584578480624807, 6)

    def test_eaccsd_koopmans(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()
        e,v = mycc.eaccsd(nroots=3, koopmans=True)
        self.assertAlmostEqual(e[0], 0.19050587834891178, 6)
        self.assertAlmostEqual(e[1], 0.2834522959728035, 6)
        self.assertAlmostEqual(e[2], 1.0213645406957659, 6)

        e,v = mycc.eaccsd(nroots=3, guess=v[:3])
        self.assertAlmostEqual(e[0], 0.19050591834647707, 6)
        self.assertAlmostEqual(e[1], 0.28345229828564883, 6)
        self.assertAlmostEqual(e[2], 1.0213649389174786, 6)

    def test_eaccsd_partition(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()

        e,v = mycc.eaccsd(nroots=3, partition='full')
        self.assertAlmostEqual(e[0], 0.18763685358256854, 6)
        self.assertAlmostEqual(e[1], 0.2796468690896177, 6)
        self.assertAlmostEqual(e[2], 0.571212134659048, 6)

        e,v = mycc.eaccsd(nroots=3, partition='mp')
        self.assertAlmostEqual(e[0], 0.19336089710651994, 6)
        self.assertAlmostEqual(e[1], 0.28722737761706013, 6)
        self.assertAlmostEqual(e[2], 0.9084868597189467 , 6)

        e,v = mycc.eaccsd(nroots=3, partition='full', left=True)

        self.assertAlmostEqual(e[0], 0.18763683639713147, 6)
        self.assertAlmostEqual(e[1], 0.2796468911587574, 6)
        self.assertAlmostEqual(e[2], 0.5712121334457637, 6)

        e,v = mycc.eaccsd(nroots=3, partition='mp', left=True)

        self.assertAlmostEqual(e[0], 0.19336088819277686, 6)
        self.assertAlmostEqual(e[1], 0.2872274525447101, 6)
        self.assertAlmostEqual(e[2], 0.9084868660163935 , 6)

    def test_eaccsd_Ta(self):
        mycc = rccsd.RCCSD(mf)
        mycc.kernel()

        e, v = mycc.eaccsd_t_star(nroots=3)
        self.assertAlmostEqual(e[0], 0.19093918198846804, 6)
        self.assertAlmostEqual(e[1], 0.2840899439275576, 6)
        self.assertAlmostEqual(e[2], 0.5233071124006111 , 6)

        e, v = mycc.eaccsd_t_star(nroots=3, left=True)
        self.assertAlmostEqual(e[0], 0.19093924648264782, 6)
        self.assertAlmostEqual(e[1], 0.2840903274078182, 6)
        self.assertAlmostEqual(e[2], 0.5233070995687656 , 6)

if __name__ == "__main__":
    print("Full Tests for RCCSD")
    unittest.main()
