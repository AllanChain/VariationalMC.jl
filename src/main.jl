# Currently following Thijssen Computational Phsics, page 375

using Statistics
using LinearAlgebra
using StructArrays
using Zygote

const Electrons = AbstractVector{Float64}
const BatchElectrons = AbstractMatrix{Float64}
const BatchLogPsi = Matrix{Float64} # Row vector
struct Params
    mo_coeff_alpha::AbstractMatrix{Float64}
    mo_coeff_beta::AbstractMatrix{Float64}
end

function dropdims_by(func::Function, a; dims)
    return dropdims(func(a, dims = dims), dims = dims)
end

function log_sgn_ψ(
    molecule::Molecule,
    params::Params,
    electrons::AbstractVector{T},
)::Tuple{T,T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    ao_α = eval_ao(molecule, electrons[:, begin:molecule.spins[1]])
    ao_β = eval_ao(molecule, electrons[:, molecule.spins[1]+1:end])
    return (params.mo_coeff_alpha * ao_α)[1, 1] * (params.mo_coeff_beta * ao_β)[1, 1], 1
    # log_ψα, sgn_ψα = logabsdet(params.mo_coeff_alpha * ao_α)
    # log_ψβ, sgn_ψβ = logabsdet(params.mo_coeff_beta * ao_β)
    # return log_ψα .* log_ψβ, sgn_ψα .* sgn_ψβ
end

function log_sgn_ψ(
    molecule::Molecule,
    params::Params,
    batch_electrons::AbstractMatrix{T},
    )::Tuple{Matrix{T},Vector{T}} where {T<:Number}
    # return broadcast(electrons -> log_sgn_ψ(params, electrons), batch_electrons)
    result = StructArray([
        log_sgn_ψ(molecule, params, electrons) for electrons in eachcol(batch_electrons)
    ])
    log_ψ, sgn_ψ = StructArrays.components(result)
    return adjoint(log_ψ), sgn_ψ
end

function log_ψ(molecule::Molecule, params::Params, electrons)
    return log_sgn_ψ(molecule, params, electrons)[1]
end

function local_energy(molecule::Molecule, params::Params, electrons::Electrons)::Float64
    return -0.5 * (
        sum(diaghessian(x -> log_ψ(molecule, params, x), electrons)) +
        sum(gradient(x -> log_ψ(molecule, params, x), electrons) .^ 2)
    )
end

function local_energy(
    molecule::Molecule,
    params::Params,
    batch_electrons::BatchElectrons,
)::Vector{Float64}
    return [
        local_energy(molecule, params, electrons) for electrons in eachcol(batch_electrons)
    ]
end

function init_walkers(
    batch_size::Integer,
    nelectrons::Integer,
    ndim::Integer = 3,
)::Matrix{Float64}
    return ones(ndim * nelectrons, batch_size) + randn(ndim * nelectrons, batch_size)
end

function mcmc_walk(
    molecule::Molecule,
    params::Params,
    electrons::BatchElectrons,
    width::Float64,
)::Tuple{BatchElectrons,Float64}
    new_walkers = electrons + randn(size(electrons)) * width
    p = exp.(2(log_ψ(molecule, params, new_walkers) - log_ψ(molecule, params, electrons)))
    cond = rand(Float64, size(p)) .< p
    # println(resize(cond, size(electrons)))
    new_walkers = ifelse.(cond, new_walkers, electrons)
    acceptance = mean(cond)
    # println("Old $electrons")
    # println("New $new_walkers")
    return new_walkers, acceptance
end


function batch_mcmc_walk(
    steps::Integer,
    molecule::Molecule,
    params::Params,
    electrons::BatchElectrons,
    width::Float64;
    adjust::Bool = false,
)
    walkers = electrons
    accum_accept = Vector{Float64}(undef, steps)
    for i = 1:steps
        walkers, acceptance = mcmc_walk(molecule, params, walkers, width)
        accum_accept[i] = acceptance
        if adjust
            if acceptance > 0.55
                width *= 1.1
            elseif acceptance < 0.5
                width *= 0.9
            end
        end
    end
    return walkers, width, accum_accept
end


function init_params(molecule::Molecule)
    nao = number_ao(molecule)
    return Params(
        rand(Float64, (molecule.spins[1], nao)),
        rand(Float64, (molecule.spins[2], nao)),
    )
end


function vmc(molecule::Molecule, batch_size::Integer)
    params = init_params(molecule)
    walkers = init_walkers(batch_size, sum(molecule.spins))
    # el = local_energy(molecule, params, walkers)
    # ev = Statistics.mean(el)
    width = 0.1
    walkers, width, _ =
        batch_mcmc_walk(500, molecule, params, walkers, width, adjust = true)

    for i = 1:50
        el = local_energy(molecule, params, walkers)
        ev = Statistics.mean(el)
        σ²e = var(el)
        ∂p_logψ = Zygote.forwarddiff(params -> log_ψ(molecule, params, walkers), params)
        el = reshape(el, (1, batch_size))
        ∂p_E =
            2(
                dropdims_by(mean, el .* ∂p_logψ, dims = 2) -
                ev .* dropdims_by(mean, ∂p_logψ, dims = 2)
            )
        println("Loop $i; Energy $ev; Variance $σ²e; Params $params")
        params = params .- ∂p_E
        walkers, width, acceptance = batch_mcmc_walk(100, molecule, params, walkers, width)
        mean_acceptance = mean(acceptance)
        if mean_acceptance > 0.55
            width *= 1.1
        elseif mean_acceptance < 0.5
            width *= 0.9
        end
    end
end
# 𝜕αlogψ = Zygote.gradient(() -> log_psi(params, walkers), Params([params]))
# println(𝜕αlogψ[params])
# println(Zygote.forwarddiff(params -> log_psi(params, walkers), params))
# println(mcmc_walk(params, walkers, 0.1))

# main(1024)
function main()
    basis = read_basis("sto-3g")
    H_basis = basis["H"]
    H₂ = Molecule([Atom(1, [0.0, 0.0, 0.0], H_basis), Atom(1, [1.4, 0.0, 0.0], H_basis)])
    vmc(H₂, 256)
end

# main()
