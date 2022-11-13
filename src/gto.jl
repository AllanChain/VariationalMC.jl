using LinearAlgebra

function sum_gaussian(l, c, a, r²)
    return exp2(l) .* (2 / π)^(3 / 4) .* sum(c .* a .^ ((2l + 3) / 4) .* exp.(-a .* r²))
end

function eval_ao(molecule::Molecule, x::Vector{T}) where {T<:Number}
    ao = Vector{Number}()
    for atom in molecule.atoms
        r = x - atom.coord
        r² = norm(r)^2
        for bas in atom.basis
            for coeff in eachcol(bas.coeff)
                gaussian_form = sum_gaussian(bas.l, coeff, bas.exp, r²)
                if bas.l == 0
                    push!(ao, gaussian_form)
                elseif bas.l == 1
                    for rᵢ in r
                        push!(ao, gaussian_form * rᵢ)
                    end
                elseif bas.l == 2
                    push!(ao, gaussian_form * r[1] * r[2])
                    push!(ao, gaussian_form * r[2] * r[3])
                    push!(ao, gaussian_form * r[3] * r[3])
                    push!(ao, gaussian_form * r[1] * r[3])
                    push!(ao, gaussian_form * r[1]^2 * r[2]^2)
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
