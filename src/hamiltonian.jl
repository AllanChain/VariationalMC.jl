function local_kinetic_energy(
    wf::WaveFunction,
    molecule::Molecule,
    electrons,
)
    return -1 / 2 * normalized_laplacian(wf, molecule, electrons)
end

function local_potential_energy(
    molecule::Molecule,
    electrons::AbstractVector{T},
)::T where {T<:Number}
    atoms = hcat([atom.coord for atom in molecule.atoms]...)
    charges = [atom.charge for atom in molecule.atoms]
    electrons = reshape(electrons, 3, :)
    r_ae = pairwise(Euclidean(), atoms, electrons, dims = 2)
    r_ee = pairwise(Euclidean(), electrons, dims = 2)
    potential_energy = -sum(charges ./ r_ae) + sum(triu(1 ./ r_ee, 1))
    if length(molecule.atoms) > 1
        r_aa = pairwise(Euclidean(), atoms, dims = 2)
        potential_energy += sum(triu(charges .* reshape(charges, 1, :) / r_aa, 1))
    end
    return potential_energy
end

function local_potential_energy(
    molecule::Molecule,
    batch_electrons::AbstractMatrix{T},
)::Vector{T} where {T<:Number}
    return [
        local_potential_energy(molecule, electrons) for
        electrons in eachcol(batch_electrons)
    ]
end

function local_energy(wf::WaveFunction, molecule::Molecule, electrons)
    el = (
        local_kinetic_energy(wf, molecule, electrons) +
        local_potential_energy(molecule, electrons)
    )
    q1, q3 = quantile(el, [0.25, 0.75])
    iqr = q3 - q1
    clamp!(el, q1 - 3 * iqr, q3 + 3 * iqr)
    return el
end

function local_energy_deriv_params(wf::WaveFunction, molecule::Molecule, electrons)
    el = local_energy(wf, molecule, electrons)
    ev = mean(el)
    ∂p_logψ = dp_log(wf, molecule, electrons)
    mean_∂p_logψ = mean_∂p(∂p_logψ)
    StructArrays.foreachfield(v -> v .*= el, ∂p_logψ)
    mean_el_∂p_logψ = mean_∂p(∂p_logψ)
    return el, 2 .* (mean_el_∂p_logψ .- ev .* mean_∂p_logψ)
end

