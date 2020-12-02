using TOML
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

function get_stencil(parsed_toml, path...; center=nothing)
    if length(path) == 0
        @assert parsed_toml isa Vector
        stencil_weights = Float64.(parse_rational.(parsed_toml))

        width = length(stencil_weights)

        if isnothing(center)
            center = div(width,2)+1
        end

        return Stencil(Tuple(stencil_weights), center=center)
    end

    return get_stencil(parsed_toml[path[1]], path[2:end]...; center=center)
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
