using LinearAlgebra
using StructArrays

export WaveFunction, update_func!, log_func, signed_log_func, dp_log, dx_log,
    normalized_laplacian, laplacian_log

abstract type WaveFunction end

function signed_log_func(
    func::WaveFunction,
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
)::Tuple{Matrix{T},Vector{T}} where {T<:Number}
    result = StructArray([
        signed_log_func(func, molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ])
    log_funcs, sgn_funcs = StructArrays.components(result)
    return adjoint(log_funcs), sgn_funcs
end

function log_func(
    func::WaveFunction,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    return signed_log_func(func, molecule, electrons)[1]
end

function log_func(
    func::WaveFunction,
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
)::Matrix{T} where {T<:Number}
    return adjoint([
        log_func(func, molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ])
end

function dp_log(
    func::WaveFunction,
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
) where {T<:Number}
    return StructArray([
        dp_log(func, molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ])
end

function normalized_laplacian(
    func::WaveFunction,
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
)::Vector{T} where {T<:Number}
    return [
        normalized_laplacian(func, molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ]
end
