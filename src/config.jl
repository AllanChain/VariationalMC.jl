import TOML

export load_config, build_molecule

recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
recursive_merge(x::AbstractVector...) = cat(x...; dims = 1)
recursive_merge(x...) = x[end]

function load_config(config_file::String)
    base_config = TOML.parsefile(joinpath(@__DIR__, "default_config.toml"))
    user_config = TOML.parsefile(config_file)
    return recursive_merge(base_config, user_config)
end

function build_molecule(config::Dict)
    system_cfg = config["system"]
    basis = read_basis(system_cfg["basis"])
    atoms = Vector{Atom}(undef, length(system_cfg["atoms"]))
    for (i, atom_cfg) in enumerate(system_cfg["atoms"])
        atoms[i] = Atom(atom_cfg["name"], atom_cfg["coord"], basis[atom_cfg["name"]])
    end
    spins = (system_cfg["spins"][1], system_cfg["spins"][2])
    return Molecule(atoms, spins)
end
