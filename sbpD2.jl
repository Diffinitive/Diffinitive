abstract type ConstantStencilOperator end

function apply!(op::ConstantStencilOperator, u::AbstractVector, v::AbstractVector, h::Real)
    N = length(v)
    cSize = closureSize(op)

    for i ∈ range(1; length=cSize)
        u[i] = apply(op.closureStencils[i], v, i)/h^2
    end

    innerStart = 1 + cSize
    innerEnd = N - cSize
    for i ∈ range(innerStart, stop=innerEnd)
        u[i] = apply(op.innerStencil, v, i)/h^2
    end

    for i ∈ range(innerEnd+1, length=cSize)
        u[i] = op.parity*apply(flip(op.closureStencils[N-i+1]), v, i)/h^2
    end
end

odd = -1
even = 1

struct D2{T} <: ConstantStencilOperator
    quadratureClosure::Vector{T}
    innerStencil::Stencil
    closureStencils::Vector{Stencil} # TBD: Should this be a tuple?
    eClosure::Vector{T}
    dClosure::Vector{T}
    parity::Int
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

    innerStencil = Stencil(r, innerStencilWeights)

    # Create boundary stencils
    boundarySize = length(d["boundary_stencils"])
    closureStencils = Vector{Stencil}()

    for i ∈ 1:boundarySize
        stencilWeights = stringToVector(Float64, d["boundary_stencils"][i])
        width = length(stencilWeights)
        r = (1-i,width-i)
        push!(closureStencils,Stencil(r, stencilWeights))
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
