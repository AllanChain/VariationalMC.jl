export SlaterJastrow

mutable struct SlaterJastrow <: WaveFunction
    slater::SlaterDetProd
    jastrow::Jastrow
end

function SlaterJastrow(molecule::Molecule)
    return SlaterJastrow(
        SlaterDetProd(molecule),
        Jastrow(rand()),
    )
end

function update_func!(
    sj::SlaterJastrow,
    params::Tuple{Matrix{T},Matrix{T},T},
) where {T<:Number}
    update_func!(sj.slater, params[1:2])
    update_func!(sj.jastrow, params[3:end])
end

function zeros_like_params(
    sj::SlaterJastrow,
)::Tuple{Matrix{Float64},Matrix{Float64},Float64}
    return (zeros_like_params(sj.slater)..., 0)
end

function signed_log_func(
    sj::SlaterJastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{T,T} where {T<:Number}
    log_slater, sgn_slater = signed_log_func(sj.slater, molecule, electrons)
    log_jastrow, sgn_jastrow = signed_log_func(sj.jastrow, molecule, electrons)
    return log_slater + log_jastrow, sgn_slater * sgn_jastrow
end

function dp_log(
    sj::SlaterJastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{Matrix{T},Matrix{T},T} where {T<:Number}
    return (
        dp_log(sj.slater, molecule, electrons)...,
        dp_log(sj.jastrow, molecule, electrons)...,
    )
end

function dx_log(
    sj::SlaterJastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Vector{T} where {T<:Number}
    return (
        dx_log(sj.slater, molecule, electrons) +
        dx_log(sj.jastrow, molecule, electrons)
    )
end

function laplacian_log(
    sj::SlaterJastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    return (
        laplacian_log(sj.slater, molecule, electrons) +
        laplacian_log(sj.jastrow, molecule, electrons)
    )
end

function normalized_laplacian(sj::SlaterJastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    dx_log_j = dx_log(sj.jastrow, molecule, electrons)
    # laplacian_log is simpler for Jastrow, normalized_laplacian is simpler for Slater
    return (
        laplacian_log(sj.jastrow, molecule, electrons) + sum(dx_log_j .^ 2) +
        2 * dot(dx_log_j, dx_log(sj.slater, molecule, electrons)) +
        normalized_laplacian(sj.slater, molecule, electrons)
    )
end
