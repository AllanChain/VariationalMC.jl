using LinearAlgebra

export log_func, signed_log_func, dp_log, dx_log, normalized_laplacian, laplacian_log

function log_func(
    func,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    return signed_log_func(func, molecule, electrons)[1]
end
