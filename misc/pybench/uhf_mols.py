from pyscf import gto, scf

mols = [
    "H 0 0 0; H 1.4 0 0",
    "Li 0 0 0; Li 5 0 0",
    "N 0 0 0; N 2 0 0",
    """
O -1.1712    0.2997    0.0000
C -0.0463   -0.5665    0.0000
C  1.2175    0.2668    0.0000
H -0.0958   -1.2120    0.8819
H -0.0952   -1.1938   -0.8946
H  2.1050   -0.3720   -0.0177
H  1.2426    0.9307   -0.8704
H  1.2616    0.9052    0.8886
H -1.1291    0.8364    0.8099
    """,
]

for atoms in mols:
    print(atoms)
    mol = gto.M(atom=atoms, basis="6-31g", unit="B")
    mf = scf.UHF(mol)
    mf.kernel()
    print(mf.mo_coeff[0, 0])
