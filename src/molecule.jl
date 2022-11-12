include("./basis.jl")

struct Atom
    coord::Vector{Number}
    basis::Basis
    charge::Number
end

struct Molecule
    atoms::Vector{Atom}
end
