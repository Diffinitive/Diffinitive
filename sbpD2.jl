struct D2{T}
    quadratureClosure::Vector{T}
    innerStencil::Stencil
    closureStencils::Vector{Stencil} # TBD: Should this be a tuple?
    eClosure::Vector{T}
    dClosure::Vector{T}
end

function closureSize(D::D2)::Int
    return length(quadratureClosure)
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
