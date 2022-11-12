import TOML

recursive_merge(x::AbstractDict...) = merge(recursive_merge, x...)
recursive_merge(x::AbstractVector...) = cat(x...; dims=1)
recursive_merge(x...) = x[end]

function load_config()
    base_config = open(joinpath(@__DIR__, "default_config.toml")) |> TOML.parsefile
    user_config = open("config.toml") |> TOML.parsefile
    return recursive_merge(base_config, user_config)
end
