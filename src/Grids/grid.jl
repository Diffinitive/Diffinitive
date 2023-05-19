"""
     Grid{T,D}

A grid with coordinates of type `T`, e.g. `SVector{3,Float64}`, and dimension
`D`. The grid can be embedded in a higher dimension in which case the number
of indices and the number of components of the coordinate vectors will be
different.

All grids are expected to behave as a grid function for the coordinates.

`Grids` is top level abstract type for grids. A grid should implement Julia's interfaces for
indexing and iteration.

## Note

Importantly a grid does not have to be an `AbstractArray`. The reason is to
allow flexible handling of special types of grids like multi-block grids, or
grids with special indexing.
"""
abstract type Grid{T,D} end

Base.ndims(::Grid{T,D}) where {T,D} = D
Base.eltype(::Type{<:Grid{T}}) where T = T

"""
    coordinate_size(g)

The lenght of the coordinate vector of `Grid` `g`.
"""
coordinate_size(::Type{<:Grid{T}}) where T = _ncomponents(T)
coordinate_size(g::Grid) = coordinate_size(typeof(g)) # TBD: Name of this function?!

"""
    component_type(g)

The type of the components of the coordinate vector of `Grid` `g`.
"""
component_type(::Type{<:Grid{T}}) where T = eltype(T)
component_type(g::Grid) = component_type(typeof(g))

"""
    refine(g::Grid, r)

The grid where `g` is refined by the factor `r`.

See also: [`coarsen`](@ref).
"""
function refine end

"""
    coarsen(g::Grid, r)

The grid where `g` is coarsened by the factor `r`.

See also: [`refine`](@ref).
"""
function coarsen end

"""
    boundary_identifiers(g::Grid)

Identifiers for all the boundaries of `g`.
"""
function boundary_identifiers end

"""
    boundary_grid(g::Grid, id::BoundaryIdentifier)

The grid for the boundary specified by `id`.
"""
function boundary_grid end
# TBD: Can we implement a version here that accepts multiple ids and grouped boundaries? Maybe we need multiblock stuff?

"""
    eval_on(g::Grid, f)

Lazy evaluation `f` on the grid. `f` can either be on the form `f(x,y,...)`
with each coordinate as an argument, or on the form `f(x̄)` taking a
coordinate vector.

For concrete array grid functions `map(f,g)` can be used instead.
"""
eval_on(g::Grid, f) = eval_on(g, f, Base.IteratorSize(g))
function eval_on(g::Grid, f, ::Base.HasShape)
    if hasmethod(f, (Any,))
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]), size(g))
    else
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]...), size(g))
    end
end

_ncomponents(::Type{<:Number}) = 1
_ncomponents(T::Type{<:SVector}) = length(T)
