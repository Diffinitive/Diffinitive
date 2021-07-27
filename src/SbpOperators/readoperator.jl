using TOML

export read_stencil_set
export get_stencil_set

export parse_stencil
export parse_rational

# The read_stencil_set and get_stencil_set functions return the freshly parsed
# toml. The generic code in these functions can't be expected to know anyhting
# about how to read different stencil sets as they may contain many different
# kinds of stencils. We should how ever add read_ and get_ functions for all
# the types of stencils we know about.
#
# After getting a stencil set the user can use parse functions to parse what
# they want from the TOML dict. I.e no more "paths".
# Functions needed:
#   * parse stencil
#   * parse rational
#
# maybe there is a better name than parse?
# Would be nice to be able to control the type in the stencil

# TODO: Control type for the stencil
# TODO: Think about naming and terminology around freshly parsed TOML.
# Vidar: What about get_stencil instead of parse_stencil for an already parsed
# toml. It matches get_stencil_set.

"""
    read_stencil_set(fn; filters)

Picks out a stencil set from the given toml file based on some filters.
If more than one set matches the filters an error is raised.

The stencil set is not parsed beyond the inital toml parse. To get usable
stencils use the `parse_stencil` functions on the fields of the stencil set.
"""
read_stencil_set(fn; filters...) = get_stencil_set(TOML.parsefile(fn); filters...)

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

"""
    parse_stencil(toml)

Accepts parsed toml and reads it as a stencil
"""
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
        throw(ArgumentError("the TOML for a stencil must be a vector of strings or a table."))
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

function parse_rational(str)
    expr = Meta.parse(replace(str, "/"=>"//"))
    return eval(:(Rational($expr)))
end

sbp_operators_path() = (@__DIR__) * "/operators/"
export sbp_operators_path
