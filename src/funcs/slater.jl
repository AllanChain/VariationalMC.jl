export SlaterDetProd

mutable struct SlaterDetProd <: WaveFunction
    mo_coeff_alpha::AbstractMatrix{Float64}
    mo_coeff_beta::AbstractMatrix{Float64}
end

function SlaterDetProd(molecule::Molecule)
    nao = number_ao(molecule)
    return SlaterDetProd(
        rand(Float64, (molecule.spins[1], nao)),
        rand(Float64, (molecule.spins[2], nao)),
    )
end

function by_αβ(func::Function, molecule::Molecule, electrons)
    result = func(molecule, electrons)
    if ndims(result) == 2
        return result[:, begin:molecule.spins[1]], result[:, molecule.spins[1]+1:end]
    elseif ndims(result) == 3
        return result[:, :, begin:molecule.spins[1]], result[:, :, molecule.spins[1]+1:end]
    else
        throw(ErrorException("Result returns unexcepted dims $(ndims(result))"))
    end
end

function update_func!(
    slater::SlaterDetProd,
    params::Tuple{Matrix{T},Matrix{T}},
) where {T<:Number}
    slater.mo_coeff_alpha += params[1]
    slater.mo_coeff_beta += params[2]
end

function signed_log_func(
    slater::SlaterDetProd,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{T,T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    log_ψα, sgn_ψα = logabsdet(slater.mo_coeff_alpha * ao_α)
    log_ψβ, sgn_ψβ = logabsdet(slater.mo_coeff_beta * ao_β)
    return log_ψα .+ log_ψβ, sgn_ψα .* sgn_ψβ
end

function dp_log(
    slater::SlaterDetProd,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{Matrix{T},Matrix{T}} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    A_α = slater.mo_coeff_alpha * ao_α
    A_β = slater.mo_coeff_beta * ao_β
    return (transpose(ao_α / A_α), transpose(ao_β / A_β))
end

function dx_log(
    slater::SlaterDetProd,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Vector{T} where {T<:Number} # shape (3 * nelec,)
    electrons = reshape(electrons, 3, :)
    # Shape (nao, nelec)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    # Shape (3, nao, nelec)
    ∇ao_α, ∇ao_β = by_αβ(eval_ao_deriv, molecule, electrons)
    A_α = slater.mo_coeff_alpha * ao_α
    A_β = slater.mo_coeff_beta * ao_β
    # Shape (1, nelec, nao)
    inv_A_α_mo = (A_α\slater.mo_coeff_alpha)[[CartesianIndex()], :, :]
    inv_A_β_mo = (A_β\slater.mo_coeff_beta)[[CartesianIndex()], :, :]
    # Dot product over dims=2. a (1, nao); b (3, nao); return (3,)
    dotdot(a, b) = dropdims(sum(a .* b, dims = 2), dims = 2)
    return reshape(
        hcat(
            dotdot.(eachslice(inv_A_α_mo, dims = 2), eachslice(∇ao_α, dims = 3))...,
            dotdot.(eachslice(inv_A_β_mo, dims = 2), eachslice(∇ao_β, dims = 3))...,
        ),
        :,
    )
end

function normalized_laplacian(
    slater::SlaterDetProd,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ∇²ao_α, ∇²ao_β = by_αβ(eval_ao_laplacian, molecule, electrons)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    A_α = slater.mo_coeff_alpha * ao_α
    A_β = slater.mo_coeff_beta * ao_β
    return tr(A_α \ slater.mo_coeff_alpha * ∇²ao_α) +
           tr(A_β \ slater.mo_coeff_beta * ∇²ao_β)
end

"""
It's just

    normalized_laplacian - sum(dx_log .^ 2)

But merged some calculations.
"""
function laplacian_log(
    slater::SlaterDetProd,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    # return normalized_laplacian(slater, molecule, electrons) - sum(
    #     dx_log(slater, molecule, electrons) .^ 2,
    # )
    electrons = reshape(electrons, 3, :)
    ∇²ao_α, ∇²ao_β = by_αβ(eval_ao_laplacian, molecule, electrons)
    ∇ao_α, ∇ao_β = by_αβ(eval_ao_deriv, molecule, electrons)
    ao_α, ao_β = by_αβ(eval_ao, molecule, electrons)
    A_α = slater.mo_coeff_alpha * ao_α
    A_β = slater.mo_coeff_beta * ao_β
    inv_A_α_mo = A_α \ slater.mo_coeff_alpha
    inv_A_β_mo = A_β \ slater.mo_coeff_beta
    part1 = tr(inv_A_α_mo * ∇²ao_α) + tr(inv_A_β_mo * ∇²ao_β)
    inv_A_α_mo = inv_A_α_mo[[CartesianIndex()], :, :]
    inv_A_β_mo = inv_A_β_mo[[CartesianIndex()], :, :]
    dotdot(a, b) = dropdims(sum(a .* b, dims = 2), dims = 2)
    part2 = sum(
        hcat(
            dotdot.(eachslice(inv_A_α_mo, dims = 2), eachslice(∇ao_α, dims = 3))...,
            dotdot.(eachslice(inv_A_β_mo, dims = 2), eachslice(∇ao_β, dims = 3))...,
        ) .^ 2,
    )
    return part1 - part2
end
