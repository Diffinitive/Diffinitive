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

struct CartesianAtlas <: Atlas
    charts::Matrix{Chart}
end

charts(a::CartesianAtlas) = a.charts
Base.size(a::CartesianAtlas) = size(a.charts)

function connections(a::CartesianAtlas)
    c = Tuple{MultiBlockBoundary, MultiBlockBoundary}[]

    N,M = size(a.charts)
    for j ∈ 1:M, i ∈ 1:N-1
        push!(c,
            (
                MultiBlockBoundary{(i,j), CartesianBoundary{1,UpperBoundary}}(),
                MultiBlockBoundary{(i+1,j), CartesianBoundary{1,LowerBoundary}}(),
            ),
        )
    end

    for i ∈ 1:N, j ∈ 1:M-1
        push!(c,
            (
                MultiBlockBoundary{(i,j), CartesianBoundary{2,UpperBoundary}}(),
                MultiBlockBoundary{(i,j+1), CartesianBoundary{2,LowerBoundary}}(),
            ),
        )
    end

    return c
end

struct UnstructuredAtlas <: Atlas
    charts::Vector{Chart}
    connections
end

charts(a::UnstructuredAtlas) = a.charts
connections(a::UnstructuredAtlas) = nothing
