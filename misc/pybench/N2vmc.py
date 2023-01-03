import time

import numpy as np
import pyqmc.api as pyq
from pyscf import gto, scf

atoms = "N 0 0 0; N 2 0 0"
nconfig = 256
mol = gto.M(atom=atoms, basis="6-31g")
mf = scf.RHF(mol)
mf.kernel()

wf = pyq.MultiplyWF(pyq.Slater(mol, mf), pyq.generate_jastrow(mol)[0])
to_opt = {
    k: np.ones_like(wf.parameters[k], dtype=bool) for k in ["wf2acoeff", "wf2bcoeff"]
}
to_opt["wf2bcoeff"][0, :] = False

coords = pyq.initial_guess(mol, nconfig)
gradient_accumulator = pyq.gradient_generator(mol, wf, to_opt)
start_time = time.time()
wf, minimization_results = pyq.line_minimization(
    wf,
    coords,
    gradient_accumulator,
    max_iterations=10,
    vmcoptions=dict(nsteps_per_block=20),
    # warmup_options=dict(nblocks=0),
    verbose=True,
)
optimize_time = time.time() - start_time
print(f"Optimize time {optimize_time}, per step {optimize_time / 10}")
