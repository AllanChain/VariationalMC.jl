include("./basis.jl")

struct Atom
    charge::Number
    coord::Vector{AbstractFloat}
    basis::Vector{Basis}
end

struct Molecule
    atoms::Vector{Atom}
end
