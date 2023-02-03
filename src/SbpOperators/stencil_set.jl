using TOML


"""
    StencilSet

A `StencilSet` contains a set of associated stencils. The stencils
are are stored in a table, and can be accesed by indexing into the `StencilSet`.
"""
struct StencilSet
    table
end
Base.getindex(set::StencilSet,I...) = set.table[I...]


"""
    read_stencil_set(filename; filters)

Creates a `StencilSet` from a TOML file based on some key-value
filters. If more than one set matches the filters an error is raised. The
table of the `StencilSet` is a parsed TOML intended for functions like
`parse_scalar` and `parse_stencil`.

The `StencilSet` table is not parsed beyond the inital TOML parse. To get usable
stencils use the `parse_stencil` functions on the fields of the stencil set.

The reason for this is that since stencil sets are intended to be very
general, and currently do not include any way to specify how to parse a given
section, the exact parsing is left to the user.

For more information see [Operator file format](@ref) in the documentation.

See also [`StencilSet`](@ref), [`sbp_operators_path`](@ref), [`get_stencil_set`](@ref), [`parse_stencil`](@ref), [`parse_scalar`](@ref), [`parse_tuple`](@ref).
"""
read_stencil_set(filename; filters...) = StencilSet(get_stencil_set(TOML.parsefile(filename); filters...))


"""
    get_stencil_set(parsed_toml; filters...)

Picks out a stencil set from an already parsed TOML based on some key-value
filters.

See also [`read_stencil_set`](@ref).
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

"""
    parse_stencil(parsed_toml)

Accepts parsed TOML and reads it as a stencil.

See also [`read_stencil_set`](@ref), [`parse_scalar`](@ref), [`parse_tuple`](@ref).
"""
function parse_stencil(parsed_toml)
    check_stencil_toml(parsed_toml)

    if parsed_toml isa Array
        weights = parse_rational.(parsed_toml)
        return CenteredStencil(weights...)
    end

    weights = parse_rational.(parsed_toml["s"])
    return Stencil(weights..., center = parsed_toml["c"])
end

"""
    parse_stencil(T, parsed_toml)

Parses the input as a stencil with element type `T`.
"""
parse_stencil(T, parsed_toml) = Stencil{T}(parse_stencil(parsed_toml))

function check_stencil_toml(parsed_toml)
    if !(parsed_toml isa Dict || parsed_toml isa Vector{String})
        throw(ArgumentError("the TOML for a stencil must be a vector of strings or a table."))
    end

    if parsed_toml isa Vector{String}
        return
    end

    if !(haskey(parsed_toml, "s") && haskey(parsed_toml, "c"))
        throw(ArgumentError("the table form of a stencil must have fields `s` and `c`."))
    end

    if !(parsed_toml["s"] isa Vector{String})
        throw(ArgumentError("a stencil must be specified as a vector of strings."))
    end

    if !(parsed_toml["c"] isa Int)
        throw(ArgumentError("the center of a stencil must be specified as an integer."))
    end
end

"""
    parse_scalar(parsed_toml)

Parse a scalar, represented as a string or a number in the TOML, and return it as a `Rational`

See also [`read_stencil_set`](@ref), [`parse_stencil`](@ref) [`parse_tuple`](@ref).
"""
function parse_scalar(parsed_toml)
    try
        return parse_rational(parsed_toml)
    catch e
        throw(ArgumentError("must be a number or a string representing a number."))
    end
end

"""
    parse_tuple(parsed_toml)

Parse an array as a tuple of scalars.

See also [`read_stencil_set`](@ref), [`parse_stencil`](@ref), [`parse_scalar`](@ref).
"""
function parse_tuple(parsed_toml)
    if !(parsed_toml isa Array)
        throw(ArgumentError("argument must be an array"))
    end
    return Tuple(parse_scalar.(parsed_toml))
end


"""
    parse_rational(parsed_toml)

Parse a string or a number as a rational.
"""
function parse_rational(parsed_toml)
    if parsed_toml isa String
        expr = Meta.parse(replace(parsed_toml, "/"=>"//"))
        return eval(:(Rational($expr)))
    else
        return Rational(parsed_toml)
    end
end

"""
    sbp_operators_path()

Calculate the path for the operators folder with included stencil sets.

See also [`StencilSet`](@ref), [`read_stencil_set`](@ref).
"""
sbp_operators_path() = (@__DIR__) * "/operators/"
