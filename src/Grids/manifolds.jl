"""
    Chart{D}

A parametrized description of a manifold or part of a manifold.
"""
struct Chart{D, PST<:ParameterSpace{D}, MT}
    mapping::MT
    parameterspace::PST
end

Base.ndims(::Chart{D}) where D = D
(c::Chart)(ξ) = c.mapping(ξ)
parameterspace(c::Chart) = c.parameterspace

"""
    jacobian(c::Chart, ξ)

The jacobian of the mapping evaluated at `ξ`. This defers to the
implementation of `jacobian` for the mapping itself. If no implementation is
available one can easily be specified for either the mapping function or the
chart itself.
```julia
c = Chart(f, ps)
jacobian(f::typeof(f), ξ) = f′(ξ)
```
or
```julia
c = Chart(f, ps)
jacobian(c::typeof(c),ξ) = f′(ξ)
```
which will both allow calling `jacobian(c,ξ)`.
"""
jacobian(c::Chart, ξ) = jacobian(c.mapping, ξ)
# TBD: Can we register a error hint for when jacobian is called with a function that doesn't have a registered jacobian?


# TBD: Should Charts, parameterspaces, Atlases, have boundary names?

"""
    Atlas

A collection of charts and their connections.
Should implement methods for `charts` and `connections`.
"""
abstract type Atlas end

"""
    charts(::Atlas)

The colloction of charts in the atlas.
"""
function charts end

"""
    connections(::Atlas)

TBD: What exactly should this return?
"""
function connections end

struct CartesianAtlas{D,C<:Chart} <: Atlas
    charts::AbstractArray{C,D}
end

charts(a::CartesianAtlas) = a.charts
Base.size(a::CartesianAtlas) = size(a.charts)

function connections(a::CartesianAtlas)
    c = Tuple{MultiBlockBoundary, MultiBlockBoundary}[]

    for d ∈ 1:ndims(charts(a))
        Is = eachslice(CartesianIndices(charts(a)); dims=d)
        for i ∈ 1:length(Is)-1 # For each interface between slices
            for jk ∈ eachindex(Is[i]) # For each block in slice
                Iᵢⱼₖ = Tuple(Is[i][jk])
                Iᵢ₊₁ⱼₖ = Tuple(Is[i+1][jk])
                push!(c,
                    (
                        MultiBlockBoundary{Iᵢⱼₖ,   CartesianBoundary{d,UpperBoundary}}(),
                        MultiBlockBoundary{Iᵢ₊₁ⱼₖ, CartesianBoundary{d,LowerBoundary}}(),
                    )
                )
            end
        end
    end

    return c
end

struct UnstructuredAtlas <: Atlas
    charts::Vector{Chart}
    connections::Vector{Tuple{MultiBlockBoundary, MultiBlockBoundary}}
end

charts(a::UnstructuredAtlas) = a.charts
connections(a::UnstructuredAtlas) = nothing
