export SlaterJastrow

mutable struct SlaterJastrow
    slater::SlaterDetProd
    jastrow::Jastrow
end

function SlaterJastrow(molecule::Molecule)
    return SlaterJastrow(
        SlaterDetProd(molecule),
        Jastrow(rand()),
    )
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
    return (
        laplacian_log(sj.jastrow, molecule, electrons) + sum(dx_log_j .^ 2) +
        2 * dot(dx_log_j, dx_log(sj.slater, molecule, electrons)) +
        normalized_laplacian(sj.slater, molecule, electrons)
    )
end
