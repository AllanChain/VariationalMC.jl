# VariationalMC.jl

Performing variational quantum Monte Carlo (VMC) in Julia.

## :warning: Warning

This is my project for the course, and is for educational purpose only. It's not, and probably will never be, ready for researches. And there will be much less updates in the future.

But I'm not stopping anyone to write VMC software based on this project, as long as the license is conformed.

## Running the example

Clone the repository and in Julia REPL, run

```julia
pkg> activate .
julia> using VariationalMC
julia> vmc(load_config("Li2.toml"; qmc_iterations=10))
```

## Configuration system

See `src/config.jl`

## More documentations and tutorials

Coming soon...
