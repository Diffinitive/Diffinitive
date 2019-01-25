abstract type ConstantStencilOperator end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Int)
    cSize = closureSize(op)
    N = length(v)

    if i ∈ range(1; length=cSize)
        @inbounds uᵢ = apply(op.closureStencils[i], v, i)/h^2
    elseif i ∈ range(N - cSize+1, length=cSize)
        @inbounds uᵢ = Int(op.parity)*apply(flip(op.closureStencils[N-i+1]), v, i)/h^2
    else
        @inbounds uᵢ = apply(op.innerStencil, v, i)/h^2
    end

    return uᵢ
end

@enum Parity begin
    odd = -1
    even = 1
end

struct D2{T,N,M,K} <: ConstantStencilOperator
    quadratureClosure::Vector{T}
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M, Stencil{T,K}}
    eClosure::Vector{T}
    dClosure::Vector{T}
    parity::Parity
end

function closureSize(D::D2)::Int
    return length(D.quadratureClosure)
end

function readOperator(D2fn, Hfn)
    d = readSectionedFile(D2fn)
    h = readSectionedFile(Hfn)

    # Create inner stencil
    innerStencilWeights = stringToVector(Float64, d["inner_stencil"][1])
    width = length(innerStencilWeights)
    r = (-div(width,2), div(width,2))

    innerStencil = Stencil(r, Tuple(innerStencilWeights))

    # Create boundary stencils
    boundarySize = length(d["boundary_stencils"])
    closureStencils = Vector{typeof(innerStencil)}() # TBD: is the the right way to get the correct type?

    for i ∈ 1:boundarySize
        stencilWeights = stringToVector(Float64, d["boundary_stencils"][i])
        width = length(stencilWeights)
        r = (1-i,width-i)
        closureStencils = (closureStencils..., Stencil(r, Tuple(stencilWeights)))
    end

    d2 = D2(
        stringToVector(Float64, h["closure"][1]),
        innerStencil,
        closureStencils,
        stringToVector(Float64, d["e"][1]),
        stringToVector(Float64, d["d1"][1]),
        even
    )

    return d2
end


function readSectionedFile(filename)::Dict{String, Vector{String}}
    f = open(filename)
    sections = Dict{String, Vector{String}}()
    currentKey = ""

    for ln ∈ eachline(f)
        if ln == "" || ln[1] == '#' # Skip comments and empty lines
            continue
        end

        if isletter(ln[1]) # Found start of new section
            if ~haskey(sections, ln)
                sections[ln] =  Vector{String}()
            end
            currentKey = ln
            continue
        end

        push!(sections[currentKey], ln)
    end

    return sections
end

function stringToVector(T::DataType, s::String)
    return T.(eval.(Meta.parse.(split(s))))
end
