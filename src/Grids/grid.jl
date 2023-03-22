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
# TODO
"""
function refine end

"""
# TODO
"""
function coarsen end

"""
# TODO
"""
function boundary_identifiers end

"""
# TODO
"""
function boundary_grid end
# TBD Can we implement a version here that accepts multiple ids and grouped boundaries? Maybe we need multiblock stuff?


# TODO: Make sure that all grids implement all of the above.


"""
TODO:

* Mention map(f,g) if you want a concrete array
"""
eval_on(g::Grid, f) = eval_on(g, f, Base.IteratorSize(g)) # TBD: Borde f vara först som i alla map, sum, och dylikt
function eval_on(g::Grid, f, ::Base.HasShape)
    if hasmethod(f, (Any,))
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]), size(g))
    else
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]...), size(g))
    end
end


# TODO: Explain how and where these are intended to be used
_ncomponents(::Type{<:Number}) = 1
_ncomponents(T::Type{<:SVector}) = length(T)
