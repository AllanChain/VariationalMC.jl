import sys

from pyscf import gto, scf

atom = sys.argv[1]
r = float(sys.argv[2])
mol = gto.M(
    atom=f"{atom} 0 0 0; {atom} {r} 0 0",
    basis="6-31g",
    unit="B",
    verbose=5,
)

mf = scf.UHF(mol)
mf.kernel()
