from pyscf import gto, scf

mol = gto.M(
    atom="Ne 0 0 0",
    basis="6-31g",
    spin=2,
    unit="B",
)

mf = scf.UHF(mol)
mf.kernel()
