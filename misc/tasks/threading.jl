using VariationalMC

@info "Warming up"
vmc(load_config("N2.toml"; qmc_iterations=1))
@info "Warmed"

@time vmc(load_config("N2.toml"; qmc_iterations=10))
