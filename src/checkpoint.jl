import JLD
import VariationalMC.optimizer: Optimizer
import VariationalMC.funcs: WaveFunction
import Printf: @sprintf

get_ckpt_name(optimizing::Bool = true) = optimizing ? "optim-ckpt-" : "eval-ckpt-"

function find_all(restore_path::AbstractString, optimizing::Bool = true)::Vector{String}
    if !ispath(restore_path)
        return Vector{String}()
    end
    return filter(s -> startswith(s, get_ckpt_name(optimizing)), readdir(restore_path))
end

function find_most_recent(restore_path::AbstractString; optimizing::Bool = true)::String
    ckpts = find_all(restore_path, optimizing)
    if length(ckpts) == 0
        if optimizing
            return ""
        end
        # If in evaluation mode and no evaluation has been performed,
        # try doing evaluation from optimization results.
        return find_most_recent(restore_path; optimizing = true)
    end
    return joinpath(restore_path, maximum(ckpts))
end

function load(
    ckpt_file::AbstractString,
)::Tuple{WaveFunction,Matrix{Float64},Optimizer,Float64}
    data = JLD.load(ckpt_file)
    return (data["wf"], data["walkers"], data["opt_state"], data["width"])
end

function smallest_n!(a, n)
    partialsort!(a, n)
    return a[1:n]
end

function save(
    save_path::AbstractString,
    iteration::Int,
    wf::WaveFunction,
    walkers::Matrix{Float64},
    opt_state::Optimizer,
    width::Float64;
    clean_old::Int = 10,
    optimizing::Bool = true,
)::String
    old_ckpts = find_all(save_path, optimizing)
    filename = @sprintf "%s%06d.jld" get_ckpt_name(optimizing) iteration
    ckpt_file = joinpath(save_path, filename)
    JLD.@save ckpt_file wf walkers opt_state width
    if length(old_ckpts) > clean_old > 0 # clean_old <= 0 will not clean
        oldest_ckpts = smallest_n!(old_ckpts, length(old_ckpts) - clean_old)
        for oldest_ckpt in oldest_ckpts
            oldest_ckpt_file = joinpath(save_path, oldest_ckpt)
            rm(oldest_ckpt_file)
        end
        @info "Removed old checkpoint $(join(oldest_ckpts, ","))"
    end
    return ckpt_file
end
