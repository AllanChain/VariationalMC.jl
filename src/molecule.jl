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

const ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
const NUC = Dict(x => i for (i, x) in enumerate(ELEMENTS))

function Atom(symbol::String, coord::Vector{Float64}, basis::Vector{Basis})
    return Atom(NUC[symbol], coord, basis)
end
