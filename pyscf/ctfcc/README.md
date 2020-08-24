# README #

### What is this repository for? ###

* This module is for parallel mole/pbc coupled cluster calculations. All integrals are stored in memory as distributed ctf tensors(wrapped by symtensor). Functions includes

* * CCSD, CCSD(T) for mole/pbc with R/U/G reference
* * EOMIP/EA for all class above
* * EOMIP/EA*, EOMIP/EA_Ta for RCCSD/KRCCSD
* * parallel GDF

### What are the prerequisites? ###

* * CTF: https://github.com/cyclops-community/ctf
* * Symtensor: https://github.com/yangcal/symtensor
* * mpi4py: https://bitbucket.org/mpi4py/

### Who do I talk to? ###

* Questions about the ctfcc and symtensor module: younggao1994@gmail.com
* Questions about compiling CTF: solomon2@illinois.edu
