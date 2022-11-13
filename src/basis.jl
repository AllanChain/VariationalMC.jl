struct Basis
    l::Int
    exp::Array{Float64}
    coeff::Matrix{Float64}
end

const SPDF_CODE = "SPDFGHIKLMNORTU"

function read_basis(basis_name::String)
    file_name = joinpath(@__DIR__, "../basis", basis_name * ".dat")
    all_basis = Dict{String,Vector{Basis}}()
    open(file_name) do f
        element = nothing
        orbital_code = nothing
        current_basis = nothing
        for line in readlines(f)
            if length(line) == 0 || startswith(line, r"#|BASIS")
                continue
            end
            if isuppercase(line[begin])
                if element !== nothing && orbital_code !== nothing
                    if !haskey(all_basis, element)
                        all_basis[element] = []
                    end
                    current_basis = hcat(current_basis...)
                    bas_exp = current_basis[1, :]
                    bas_coeff = transpose(current_basis[2:end, :])

                    if length(orbital_code) == 1
                        orbital = findfirst(isequal(orbital_code[begin]), SPDF_CODE)
                        if orbital === nothing
                            throw(ErrorException("Orbital code $orbital_code is invalid"))
                        end
                        orbital -= 1
                        push!(
                            all_basis[element],
                            Basis(orbital, bas_exp, bas_coeff),
                        )
                    elseif orbital_code == "SP"
                        push!(
                            all_basis[element],
                            Basis(0, bas_exp, bas_coeff[:, 1:1]),
                        )
                        push!(
                            all_basis[element],
                            Basis(1, bas_exp, bas_coeff[:, 2:2]),
                        )
                    else
                        throw(ErrorException("Orbital code $orbital_code is invalid"))
                    end
                end
                if !startswith(line, "END")
                    element, orbital_code = split(line)
                    current_basis = Array{Float64}[]
                end
            else
                if current_basis === nothing
                    throw(
                        ErrorException(
                            "Invalid basis file syntax while parsing $file_name",
                        ),
                    )
                end
                current_basis =
                    push!(current_basis, [parse(Float64, x) for x in split(line)])
            end
        end
    end
    for (atom, _) in all_basis
        sort!(all_basis[atom], by = bas -> bas.l)
    end
    return all_basis
end
