export Atom, Molecule

struct Atom
    charge::Number
    coord::Vector{Float64}
    basis::Vector{Basis}
end

struct Molecule
    atoms::Vector{Atom}
    spins::Tuple{Int,Int}
end

function Molecule(atoms::Vector{Atom})
    total_electrons = sum(atom.charge for atom in atoms)
    if isodd(total_electrons)
        throw(
            ErrorException(
                "Cannot auto assign spins for total electron number $total_electrons. " *
                "Consider using Molecule(atoms, (spin_alpha, spin_beta)) instead.",
            ),
        )
    end
    half_electrons = total_electrons ÷ 2
    return Molecule(atoms, (half_electrons, half_electrons))
end
