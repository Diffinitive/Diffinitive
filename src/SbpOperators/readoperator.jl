using TOML
function read_D2_operator(fn; order)
    operators = TOML.parsefile(fn)["order$order"]
    D2 = operators["D2"]
    H = operators["H"]
    e = operators["e"]
    d1 = operators["d1"]

    # Create inner stencil
    innerStencilWeights = toml_string_array_to_tuple(Float64, D2["inner_stencil"])

    width = length(innerStencilWeights)
    r = (-div(width,2), div(width,2))

    innerStencil = Stencil(r, innerStencilWeights)

    # Create boundary stencils
    boundarySize = length(D2["closure_stencils"])
    closureStencils = Vector{typeof(innerStencil)}() # TBD: is the the right way to get the correct type?

    for i ∈ 1:boundarySize
        stencilWeights = toml_string_array_to_tuple(Float64, D2["closure_stencils"][i])
        width = length(stencilWeights)
        r = (1-i,width-i)
        closureStencils = (closureStencils..., Stencil(r, stencilWeights))
    end

    quadratureClosure = pad_tuple(toml_string_array_to_tuple(Float64, H["closure"]), boundarySize)
    eClosure = Stencil((0,boundarySize-1), pad_tuple(toml_string_array_to_tuple(Float64, e["closure"]), boundarySize))
    dClosure = Stencil((0,boundarySize-1), pad_tuple(toml_string_array_to_tuple(Float64, d1["closure"]), boundarySize))

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
