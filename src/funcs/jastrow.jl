export Jastrow

mutable struct Jastrow <: WaveFunction
    b::Float64
end

function split_αβ(
    molecule::Molecule,
    electrons::AbstractMatrix{T},
) where {T<:Number}
    return (electrons[:, begin:molecule.spins[1]], electrons[:, molecule.spins[1]+1:end])
end

function update_func!(
    jastrow::Jastrow,
    params::Tuple{T},
) where {T<:Number}
    jastrow.b += params[1]
end

function signed_log_func(
    jastrow::Jastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{T,T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    result = 0
    eα, eβ = split_αβ(molecule, electrons)
    # parallel pairs
    for i = 1:size(eα, 2)
        for j = 1:i-1
            r_ij = norm(eα[:, i] - eα[:, j])
            result += 1 / 4 * r_ij / (1 + jastrow.b * r_ij)
        end
    end
    for i = 1:size(eβ, 2)
        for j = 1:i-1
            r_ij = norm(eβ[:, i] - eβ[:, j])
            result += 1 / 4 * r_ij / (1 + jastrow.b * r_ij)
        end
    end
    # antiparallel pairs
    for i = 1:size(eα, 2)
        for j = 1:size(eβ, 2)
            r_ij = norm(eα[:, i] - eβ[:, j])
            result += 1 / 2 * r_ij / (1 + jastrow.b * r_ij)
        end
    end
    return result, 1
end

function dp_log(
    jastrow::Jastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Tuple{T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    result = 0
    eα, eβ = split_αβ(molecule, electrons)
    # parallel pairs
    for i = 1:size(eα, 2)
        for j = 1:i-1
            r_ij = norm(eα[:, i] - eα[:, j])
            result += -1 / 4 * r_ij^2 / (1 + jastrow.b * r_ij)^2
        end
    end
    for i = 1:size(eβ, 2)
        for j = 1:i-1
            r_ij = norm(eβ[:, i] - eβ[:, j])
            result += -1 / 4 * r_ij^2 / (1 + jastrow.b * r_ij)^2
        end
    end
    # antiparallel pairs
    for i = 1:size(eα, 2)
        for j = 1:size(eβ, 2)
            r_ij = norm(eα[:, i] - eβ[:, j])
            result += -1 / 2 * r_ij^2 / (1 + jastrow.b * r_ij)^2
        end
    end
    return (result,)
end

function dx_log(
    jastrow::Jastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::Vector{T} where {T<:Number}
    electrons = reshape(electrons, 3, :)
    eα, eβ = split_αβ(molecule, electrons)
    result = zeros(T, size(electrons))
    for i = 1:size(eα, 2)
        for j = 1:size(eα, 2)
            if j == i
                continue
            end
            x_ij = eα[:, i] - eα[:, j]
            r_ij = norm(x_ij)
            result[:, i] += 1 / 4 * x_ij / r_ij / (1 + jastrow.b * r_ij)^2
        end
        for j = 1:size(eβ, 2)
            x_ij = eα[:, i] - eβ[:, j]
            r_ij = norm(x_ij)
            result[:, i] += 1 / 2 * x_ij / r_ij / (1 + jastrow.b * r_ij)^2
        end
    end
    for i = 1:size(eβ, 2)
        for j = 1:size(eβ, 2)
            if j == i
                continue
            end
            x_ij = eβ[:, i] - eβ[:, j]
            r_ij = norm(x_ij)
            result[:, i+molecule.spins[1]] += 1 / 4 * x_ij / r_ij / (1 + jastrow.b * r_ij)^2
        end
        for j = 1:size(eα, 2)
            x_ij = eβ[:, i] - eα[:, j]
            r_ij = norm(x_ij)
            result[:, i+molecule.spins[1]] += 1 / 2 * x_ij / r_ij / (1 + jastrow.b * r_ij)^2
        end
    end
    return reshape(result, :)
end


function laplacian_log(
    jastrow::Jastrow,
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    electrons = reshape(electrons, 3, :)
    eα, eβ = split_αβ(molecule, electrons)
    result = 0
    for i = 1:size(eα, 2)
        for j = 1:size(eα, 2)
            if j == i
                continue
            end
            r_ij = norm(eα[:, i] - eα[:, j])
            result += 1 / 2 / r_ij / (1 + jastrow.b * r_ij)^3
        end
        for j = 1:size(eβ, 2)
            r_ij = norm(eα[:, i] - eβ[:, j])
            result += 1 / r_ij / (1 + jastrow.b * r_ij)^3
        end
    end
    for i = 1:size(eβ, 2)
        for j = 1:size(eβ, 2)
            if j == i
                continue
            end
            r_ij = norm(eβ[:, i] - eβ[:, j])
            result += 1 / 2 / r_ij / (1 + jastrow.b * r_ij)^3
        end
        for j = 1:size(eα, 2)
            r_ij = norm(eβ[:, i] - eα[:, j])
            result += 1 / r_ij / (1 + jastrow.b * r_ij)^3
        end
    end
    return result
end
