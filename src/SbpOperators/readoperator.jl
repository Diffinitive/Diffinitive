using TOML

export read_stencil_set
export get_stencil_set

export parse_stencil

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
    eClosure = Stencil(pad_tuple(toml_string_array_to_tuple(Float64, e["closure"]), boundarySize)..., center=1)
    dClosure = Stencil(pad_tuple(toml_string_array_to_tuple(Float64, d1["closure"]), boundarySize)..., center=1)

    q_tuple = pad_tuple(toml_string_array_to_tuple(Float64, H["closure"]), boundarySize)
    quadratureClosure = Vector{typeof(innerStencil)}()
    for i ∈ 1:boundarySize
        quadratureClosure = (quadratureClosure..., Stencil(q_tuple[i], center=1))
    end

    d2 = SbpOperators.D2(
        innerStencil,
        closureStencils,
        eClosure,
        dClosure,
        quadratureClosure,
        even
    )

    return d2
end

"""
    read_stencil_set(fn, filter_pairs::Vararg{Pair})

Picks out a stencil set from the given toml file based on some filters.
If more than one set matches the filters an error is raised.

The stencil set is not parsed beyond the inital toml parse. To get usable
stencils use the `parse_stencil` functions on the fields of the stencil set.
"""
read_stencil_set(fn, filter_pairs::Vararg{Pair}) = get_stencil_set(TOML.parsefile(fn), filter_pairs...)

"""
    get_stencil_set(parsed_toml; filters...)

Same as `read_stencil_set` but works on already parsed TOML.
"""
function get_stencil_set(parsed_toml; filters...)
    matches = findall(parsed_toml["stencil_set"]) do set
        for (key, val) ∈ filters
            if set[string(key)] != val
                return false
            end
        end

        return true
    end

    if length(matches) != 1
        throw(ArgumentError("filters must pick out a single stencil set"))
    end

    i = matches[1]
    return parsed_toml["stencil_set"][i]
end

function parse_stencil(toml)
    check_stencil_toml(toml)

    if toml isa Array
        weights = Float64.(parse_rational.(toml))
        return CenteredStencil(weights...)
    end

    weights = Float64.(parse_rational.(toml["s"]))
    return Stencil(weights..., center = toml["c"])
end

function check_stencil_toml(toml)
    if !(toml isa Dict || toml isa Vector{String})
        throw(ArgumentError("the TOML for a stecil must be a vector of strings or a table."))
    end

    if toml isa Vector{String}
        return
    end

    if !(haskey(toml, "s") && haskey(toml, "c"))
        throw(ArgumentError("the table form of a stencil must have fields `s` and `c`."))
    end

    if !(toml["s"] isa Vector{String})
        throw(ArgumentError("a stencil must be specified as a vector of strings."))
    end

    if !(toml["c"] isa Int)
        throw(ArgumentError("the center of a stencil must be specified as an integer."))
    end
end

"""
    read_stencil(fn, path...; [center])

Read a stencil at `path` from the file with name `fn`.
If a center is specified the given element of the stecil is set as the center.

See also: [`read_stencils`](@ref), [`read_tuple`](@ref), [`get_stencil`](@ref).

# Examples
```
read_stencil(sbp_operators_path()*"standard_diagonal.toml", "order2", "D2", "inner_stencil")
read_stencil(sbp_operators_path()*"standard_diagonal.toml", "order2", "d1", "closure"; center=1)
```
"""
read_stencil(fn, path...; center=nothing) = get_stencil(TOML.parsefile(fn), path...; center=center)

"""
    read_stencils(fn, path...; centers)

Read stencils at `path` from the file `fn`.
Centers of the stencils are specified as a tuple or array in `centers`.

See also: [`read_stencil`](@ref), [`read_tuple`](@ref), [`get_stencils`](@ref).
"""
read_stencils(fn, path...; centers) = get_stencils(TOML.parsefile(fn), path...; centers=centers)

"""
    read_tuple(fn, path...)

Read tuple at `path` from the file `fn`.

See also: [`read_stencil`](@ref), [`read_stencils`](@ref), [`get_tuple`](@ref).
"""
read_tuple(fn, path...) = get_tuple(TOML.parsefile(fn), path...)

"""
    get_stencil(parsed_toml, path...; center=nothing)

Same as [`read_stencil`](@ref)) but takes already parsed toml.
"""
get_stencil(parsed_toml, path...; center=nothing) = get_stencil(parsed_toml[path[1]], path[2:end]...; center=center)
function get_stencil(parsed_toml; center=nothing)
    @assert parsed_toml isa Vector{String}
    stencil_weights = Float64.(parse_rational.(parsed_toml))

    width = length(stencil_weights)

    if isnothing(center)
        center = div(width,2)+1
    end

    return Stencil(stencil_weights..., center=center)
end

"""
    get_stencils(parsed_toml, path...; centers)

Same as [`read_stencils`](@ref)) but takes already parsed toml.
"""
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

"""
    get_tuple(parsed_toml, path...)

Same as [`read_tuple`](@ref)) but takes already parsed toml.
"""
get_tuple(parsed_toml, path...) = get_tuple(parsed_toml[path[1]], path[2:end]...)
function get_tuple(parsed_toml)
    @assert parsed_toml isa Vector{String}
    t = Tuple(Float64.(parse_rational.(parsed_toml)))
    return t
end

function get_rationals()
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
