using QMC
using QMC.molecule

function main(steps::Int)
    basis = read_basis("6-31g")
    H_basis = basis["H"]
    H₂ = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis), Atom(1, [1.4, 0.0, 0.0], H_basis)])
    vmc(H₂, 1024, steps = steps)
end
