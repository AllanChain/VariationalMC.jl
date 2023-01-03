using VariationalMC
using VariationalMC.funcs
import VariationalMC.checkpoint
using LinearAlgebra

function get_similarity(path)
    wf = checkpoint.load(path)[1]
    # mo_coeff_alpha = reshape(wf.mo_coeff_alpha, :)
    # mo_coeff_beta = reshape(wf.mo_coeff_beta, :)
    if isa(wf, SlaterJastrow)
        wf = wf.slater
    end
    nelec = size(wf.mo_coeff_alpha, 1)
    found_pairs = Vector{Tuple{Int,Int}}()
    coss = Vector{Float64}()
    while length(found_pairs) < nelec
        cos_x = 0
        found_pair = (0, 0)
        for i = 1:nelec
            for j = 1:nelec
                if (i, j) in found_pairs
                    continue
                end
                cos_x1 =
                    dot(wf.mo_coeff_alpha[i, :], wf.mo_coeff_beta[j, :]) /
                    norm(wf.mo_coeff_alpha[i, :]) / norm(wf.mo_coeff_beta[j, :])
                if abs(cos_x1) > abs(cos_x)
                    cos_x = cos_x1
                    found_pair = (i, j)
                end
            end
        end
        push!(found_pairs, found_pair)
        push!(coss, cos_x)
    end
    return sum(abs.(coss)) / nelec
end

@info get_similarity("data/H2-slater/optim-ckpt-000500.jld")
@info get_similarity("data/Li2-slater/optim-ckpt-003000.jld")
@info get_similarity("data/N2-slater-2048/optim-ckpt-003000.jld")
@info get_similarity("data/ethanol-slater/optim-ckpt-004000.jld")
@info get_similarity("data/H2-sj/optim-ckpt-000500.jld")
@info get_similarity("data/Li2-sj/optim-ckpt-003000.jld")
@info get_similarity("data/N2-sj-2048/optim-ckpt-003000.jld")
@info get_similarity("data/ethanol-sj/optim-ckpt-004000.jld")
