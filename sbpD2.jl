abstract type ConstantStencilOperator end

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Lower})
    return @inbounds apply(op.closureStencils[Int(i)], v, Int(i))/h^2
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Interior})
    return @inbounds apply(op.innerStencil, v, Int(i))/h^2
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Upper})
    N = length(v)
    return @inbounds Int(op.parity)*apply(flip(op.closureStencils[N-Int(i)+1]), v, Int(i))/h^2
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, index::Index{Unknown})
    cSize = closureSize(op)
    N = length(v)

    i = Int(index)

    if 0 < i <= cSize
        return apply(op, h, v, Index{Lower}(i))
    elseif cSize < i <= N-cSize
        return apply(op, h, v, Index{Interior}(i))
    elseif N-cSize < i <= N
        return apply(op, h, v, Index{Upper}(i))
    else
        error("Bounds error") # TODO: Make this more standard
    end
end

# Wrapper functions for using regular indecies without specifying regions
@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Int)
    return apply(op, h, v, Index{Unknown}(i))
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
    innerStencilWeights = stringToTuple(Float64, d["inner_stencil"][1])
    width = length(innerStencilWeights)
    r = (-div(width,2), div(width,2))

    innerStencil = Stencil(r, innerStencilWeights)

    # Create boundary stencils
    boundarySize = length(d["boundary_stencils"])
    closureStencils = Vector{typeof(innerStencil)}() # TBD: is the the right way to get the correct type?

    for i ∈ 1:boundarySize
        stencilWeights = stringToTuple(Float64, d["boundary_stencils"][i])
        width = length(stencilWeights)
        r = (1-i,width-i)
        closureStencils = (closureStencils..., Stencil(r, stencilWeights))
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

function stringToTuple(T::DataType, s::String)
    return Tuple(stringToVector(T,s))
end

function stringToVector(T::DataType, s::String)
    return T.(eval.(Meta.parse.(split(s))))
end
