"""
     Grid{T,D}

The top level type for grids.

TODO:
Should implement
 * interfaces for iteration and indexing
"""
abstract type Grid{T,D} end

Base.ndims(::Grid{T,D}) where {T,D} = D
Base.eltype(::Type{<:Grid{T}}) where T = T
target_manifold_dim(::Grid{T}) where T = _ncomponents(T) # TBD: Name of this function?!
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
