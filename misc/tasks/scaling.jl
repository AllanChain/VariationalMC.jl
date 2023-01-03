using VariationalMC

function warmup_and_run(config_file; kwargs...)
    @info "Warming up"
    vmc(load_config(config_file; qmc_iterations=1, kwargs...))
    @info "Warmed"

    @time vmc(load_config(config_file; kwargs...))
end

for n = 4:4:32
    @info "Processng H$n"
    warmup_and_run("scaling/H$n.toml")
    warmup_and_run("scaling/H$n.toml"; qmc_ansatz = "slater-jastrow")
end
