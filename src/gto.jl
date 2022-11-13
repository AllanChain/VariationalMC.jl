using LinearAlgebra

function sum_gaussian(l, c, a, r²)
    return exp2(l) .* (2 / π)^(3/ 4) .* sum(c .* a .^ ((2l + 3) / 4) .* exp.(-a .* r²))
end

function eval_ao(molecule::Molecule, x::Vector{T}) where {T<:Number}
    ao = Vector{Number}()
    for atom in molecule.atoms
        r = x - atom.coord
        r² = norm(r)^2
        for bas in atom.basis
            for coeff in eachcol(bas.coeff)
                if bas.l == 0
                    push!(ao, sum_gaussian(bas.l, coeff, bas.exp, r²))
                elseif bas.l === 1
                    for rᵢ in r
                        push!(ao, sum_gaussian(bas.l, coeff, bas.exp, r²) * rᵢ)
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
