import JLD
import VariationalMC.optimizer: Optimizer
import VariationalMC.funcs: WaveFunction
import Printf: @sprintf

const CKPT_NAME = "vmcjl_ckpt"

function find_all(restore_path::AbstractString)::Vector{String}
    if !ispath(restore_path)
        return Vector{String}()
    end
    return filter(s -> startswith(s, CKPT_NAME), readdir(restore_path))
end

function find_most_recent(restore_path::AbstractString)::String
    ckpts = find_all(restore_path)
    if length(ckpts) == 0
        return ""
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
)::String
    old_ckpts = find_all(save_path)
    filename = @sprintf "%s%06d.jld" CKPT_NAME iteration
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
