"""
     Grid{T,D}

A grid with coordinates of type `T`, e.g. `SVector{3,Float64}`, and dimension
`D`. The grid can be embedded in a higher dimension in which case the number
of indices and the number of components of the coordinatevectors will be
different.

`Grids` is top level abstract type for grids. A grid should implement Julia's interfaces for
indexing and iteration.

## Note

Importantly a grid does not have to be an `AbstractArray`. The reason is to
allow flexible handling of special types of grids like multiblock-grids, or
grids with special indexing.
"""
abstract type Grid{T,D} end

Base.ndims(::Grid{T,D}) where {T,D} = D
Base.eltype(::Type{<:Grid{T}}) where T = T
coordinate_size(::Grid{T}) where T = _ncomponents(T) # TBD: Name of this function?!
component_type(::Grid{T}) where T = eltype(T)

"""
    refine(g::Grid, r)

`g` refined by the factor `r`.

See also: [`coarsen`](@ref).
"""
function refine end

"""
    coarsen(g::Grid, r)

`g` coarsened by the factor `r`.

See also: [`refine`](@ref).
"""
function coarsen end

"""
    boundary_identifiers(g::Grid)

Identifiers for all the boundaries of `g`.
"""
function boundary_identifiers end

"""
    boundary_grid(g::Grid, bid::BoundaryIdentifier)

The grid for the specified boundary.
"""
function boundary_grid end
# TBD: Can we implement a version here that accepts multiple ids and grouped boundaries? Maybe we need multiblock stuff?


# TODO: Make sure that all grids implement all of the above.


"""
    eval_on(g::Grid, f)

Lazy evaluation `f` on the grid. `f` can either be on the form `f(x,y,...)`
with each coordinate as an argument, or on the form `f(x̄)` taking a
coordinate vector.

TODO: Mention map(f,g) if you want a concrete array
"""
eval_on(g::Grid, f) = eval_on(g, f, Base.IteratorSize(g)) # TBD: Borde f vara först som i alla map, sum, och dylikt
function eval_on(g::Grid, f, ::Base.HasShape)
    if hasmethod(f, (Any,))
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]), size(g))
    else
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]...), size(g))
    end
end
# TBD: How does `eval_on` relate to `map`. Should the be closer in name?


# TODO: Explain how and where these are intended to be used
_ncomponents(::Type{<:Number}) = 1
_ncomponents(T::Type{<:SVector}) = length(T)
