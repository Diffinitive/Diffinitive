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
    innerStencilWeights = stringToVector(Float64, d["inner_stencil"])
    width = length(innerStencilWeights)
    r = (-width//2, width//2)
    innerStencil = Stencil(r, innerStencilWeights)

    # Create boundary stencils
    boundarySize = length(d["boundary_stencils"])
    closureStencils = Vector{Stencil}()
    for i ∈ 1:boundarySize
        stencilWeights = stringToVector(Float64, d["boundary_stencils"][i])

    end

    d2 = D2(
        stringToVector(Float64, h["closure"]),
        innerStencil,
        closureStencils,
        stringToVector(Float64, d["e"]),
        stringToVector(Float64, d["d1"]),
    )

    # Return d2!

    return nothing
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

function stringToVector(T::DataType, s::String; delimiter = " ")
    return parse(T, split(s, delimiter))
end
