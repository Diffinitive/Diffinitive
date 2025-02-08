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

boundary_identifiers(c::Chart) = boundary_identifiers(parameterspace(c))


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

Collection of pairs of multiblock boundary identifiers.
"""
function connections end


"""
    CartesianAtlas{D,C<:Chart,AT<:AbstractArray{C,D}} <: Atlas

An atlas where the charts are arranged and connected like an array.
"""
struct CartesianAtlas{D,C<:Chart,AT<:AbstractArray{C,D}} <: Atlas
    charts::AT
end

charts(a::CartesianAtlas) = a.charts

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

"""
    boundary_identifiers(a::CartesianAtlas)

All non-connected boundaries of the charts of `a`.
"""
function boundary_identifiers(a::CartesianAtlas)
    bs = MultiBlockBoundary[]

    for d ∈ 1:ndims(charts(a))
        Is = eachslice(CartesianIndices(charts(a)); dims=d)

        for (i,b) ∈ ((1,LowerBoundary),(length(Is),UpperBoundary)) # For first and last slice
            for jk ∈ eachindex(Is[i]) # For each block in slice
                Iᵢⱼₖ = Tuple(Is[i][jk])
                push!(bs,
                    MultiBlockBoundary{Iᵢⱼₖ,   CartesianBoundary{d,b}}(),
                )
            end
        end
    end

    return bs
end


"""
    UnstructuredAtlas{C<:Chart, CN<:Tuple{MultiBlockBoundary,MultiBlockBoundary}, ...} <: Atlas

An atlas with connections determined by a vector `MultiBlockBoundary` pairs.
"""
struct UnstructuredAtlas{C<:Chart, CN<:Tuple{MultiBlockBoundary,MultiBlockBoundary}, CV<:AbstractVector{C}, CNV<:AbstractVector{CN}} <: Atlas
    charts::CV
    connections::CNV
end

charts(a::UnstructuredAtlas) = a.charts
connections(a::UnstructuredAtlas) = a.connections

"""
    boundary_identifiers(a::UnstructuredAtlas)

All non-connected boundaries of the charts of `a`.
"""
function boundary_identifiers(a::UnstructuredAtlas)
    bs = MultiBlockBoundary[]

    for (i,c) ∈ enumerate(charts(a))
        for b ∈ boundary_identifiers(c)
            mbb = MultiBlockBoundary{i,typeof(b)}()

            if !any(cn->mbb∈cn, connections(a))
                push!(bs, mbb)
            end
        end
    end

    return bs
end
