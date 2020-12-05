using TOML

export read_D2_operator
export read_stencil
export read_stencils
export read_tuple

export get_stencil
export get_stencils
export get_tuple


function read_D2_operator(fn; order)
    operators = TOML.parsefile(fn)["order$order"]
    D2 = operators["D2"]
    H = operators["H"]
    e = operators["e"]
    d1 = operators["d1"]

    # Create inner stencil
    innerStencil = get_stencil(operators, "D2", "inner_stencil")

    # Create boundary stencils
    boundarySize = length(D2["closure_stencils"])
    closureStencils = Vector{typeof(innerStencil)}() # TBD: is the the right way to get the correct type?

    for i ∈ 1:boundarySize
        closureStencils = (closureStencils..., get_stencil(operators, "D2", "closure_stencils", i; center=i))
    end

    # TODO: Get rid of the padding here. Any padding should be handled by the consturctor accepting the stencils.
    quadratureClosure = pad_tuple(toml_string_array_to_tuple(Float64, H["closure"]), boundarySize)
    eClosure = Stencil(pad_tuple(toml_string_array_to_tuple(Float64, e["closure"]), boundarySize), center=1)
    dClosure = Stencil(pad_tuple(toml_string_array_to_tuple(Float64, d1["closure"]), boundarySize), center=1)

    d2 = SbpOperators.D2(
        quadratureClosure,
        innerStencil,
        closureStencils,
        eClosure,
        dClosure,
        even
    )

    return d2
end


"""
    read_stencil(fn, path...; [center])

Read a stencil at `path` from the file with name `fn`.
If a center is specified the given element of the stecil is set as the center.

# Examples
```
read_stencil(sbp_operators_path()*"standard_diagonal.toml", "order2", "D2", "inner_stencil")
read_stencil(sbp_operators_path()*"standard_diagonal.toml", "order2", "d1", "closure"; center=1)
```
"""
read_stencil(fn, path...; center=nothing) = get_stencil(TOML.parsefile(fn), path...; center=center)
read_stencils(fn, path...; centers=nothing) = get_stencils(TOML.parsefile(fn), path...; centers=centers)
read_tuple(fn, path...) = get_tuple(TOML.parsefile(fn), path...)

get_stencil(parsed_toml, path...; center=nothing) = get_stencil(parsed_toml[path[1]], path[2:end]...; center=center)
function get_stencil(parsed_toml; center=nothing)
    @assert parsed_toml isa Vector{String}
    stencil_weights = Float64.(parse_rational.(parsed_toml))

    width = length(stencil_weights)

    if isnothing(center)
        center = div(width,2)+1
    end

    return Stencil(Tuple(stencil_weights), center=center)
end

get_stencils(parsed_toml, path...; centers) = get_stencils(parsed_toml[path[1]], path[2:end]...; centers=centers)
function get_stencils(parsed_toml; centers)
    @assert parsed_toml isa Vector{Vector{String}}
    @assert length(centers) == length(parsed_toml)

    stencils = ()
    for i ∈ 1:length(parsed_toml)
        stencil = get_stencil(parsed_toml[i], center = centers[i])
        stencils = (stencils..., stencil)
    end

    return stencils
end

get_tuple(parsed_toml, path...) = get_tuple(parsed_toml[path[1]], path[2:end]...)
function get_tuple(parsed_toml)
    @assert parsed_toml isa Vector{String}
    t = Tuple(Float64.(parse_rational.(parsed_toml)))
    return t
end

# TODO: Probably should be deleted once we have gotten rid of read_D2_operator()
function toml_string_array_to_tuple(::Type{T}, arr::AbstractVector{String}) where T
    return Tuple(T.(parse_rational.(arr)))
end

function parse_rational(str)
    expr = Meta.parse(replace(str, "/"=>"//"))
    return eval(:(Rational($expr)))
end

function pad_tuple(t::NTuple{N, T}, n::Integer) where {N,T}
    if N >= n
        return t
    else
        return pad_tuple((t..., zero(T)), n)
    end
end

sbp_operators_path() = (@__DIR__) * "/operators/"
export sbp_operators_path
