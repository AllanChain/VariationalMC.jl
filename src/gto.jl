using LinearAlgebra

function double_factorial(n::Int)
    if n < 2
        return 1
    end
    return n * double_factorial(n - 2)
end

raw"""
    sum_gaussian(lmn, c, a, r²)

Compute the gaussian summation part for atomic orbitals.

The basic form we want to calculate is
```math
\sum_i c_i x^l y^m z^n g_i(\alpha_i, r)
```

where the normalized gaussian function looks like
```math
(2a/π)^{3/4} (4a)^{(l+m+n)/2} exp(-a r^2)
```
Note the denominator
```math
\sqrt{(2l-1)!!(2m-1)!!(2n-1)!!}
```
is not included. It's not suitable for spherical terms like x^2-y^2.

# Arguments

- `j`: the sum for x, y, z exponents, i.e. l+m+n
- `c`: the vector of coefficients
- `a`: the vector of exponents
- `r²`: the squared distance between the atom and electron
"""
function sum_gaussian(j, c, a, r²)
    return exp2(j) .* (2 / π)^(3 / 4) .*
           sum(c .* a .^ (j / 2 + 3 / 4) .* exp.(-a .* r²))
end

function eval_ao(molecule::Molecule, x::AbstractMatrix{T}) where {T<:Number}
    return hcat([eval_ao(molecule, x1) for x1 in eachcol(x)]...)
end

function eval_ao(molecule::Molecule, x::AbstractVector{T}) where {T<:Number}
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
                    push!(ao, gaussian_form * (3 * r[3]^2 - r²) / 2 / sqrt(3))
                    push!(ao, gaussian_form * r[1] * r[3])
                    push!(ao, gaussian_form * (r[1]^2 - r[2]^2) / 2)
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

function number_ao(molecule::Molecule)::Int
    nao = 0
    for atom in molecule.atoms
        for bas in atom.basis
            nao += 2 * bas.l + 1
        end
    end
    return nao
end

function sum_gaussian_deriv_n(n, l, c, a, r²)
    return exp2(l) * (2 / π)^(3 / 4) *
           sum(c .* a .^ (n + (2l + 3) / 4) .* exp.(-a .* r²))
end

function eval_ao_laplacian(molecule::Molecule, x::AbstractMatrix{T}) where {T<:Number}
    return hcat([eval_ao_laplacian(molecule, x1) for x1 in eachcol(x)]...)
end

function eval_ao_laplacian(molecule::Molecule, x::AbstractVector{T}) where {T<:Number}
    ao = Vector{Number}()
    for atom in molecule.atoms
        r = x - atom.coord
        r² = norm(r)^2
        for bas in atom.basis
            for coeff in eachcol(bas.coeff)
                d1g_term = sum_gaussian_deriv_n(1, bas.l, coeff, bas.exp, r²)
                d2g_term = sum_gaussian_deriv_n(2, bas.l, coeff, bas.exp, r²)
                if bas.l == 0
                    push!(ao, -6 * d1g_term + 4 * d2g_term * sum(r .^ 2))
                elseif bas.l == 1
                    for x in r
                        push!(ao, -10x * d1g_term + 4x * sum(r .^ 2) * d2g_term)
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
