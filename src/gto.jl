using LinearAlgebra

function sum_gaussian(c, a, r)
    return sum(c .* exp(-a .* r^2))
end

function eval_ao(molecule::Molecule, x::Vector{Number})
    ao = Vector{Number}()
    for atom in molecule.atoms
        r = norm(atom.coord - x)
        for bas in atom.basis
            for coeff in eachcol(bas.coeff)
                if bas.l == 0
                    push!(ao, sum_gaussian(coeff, bas.exp, r))
                elseif bas.l === 1
                    for rᵢ in r
                        push!(ao, sum_gaussian(coeff, bas.exp, r) * rᵢ)
                    end
                else
                    throw(
                        ErrorException(
                            "Orbitals with angular momentum $(bas.l) is not supported",
                        ),
                    )
                end
            end
        end
    end
    return ao
end
