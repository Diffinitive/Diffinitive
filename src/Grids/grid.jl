"""
     Grid{T,D}

The top level type for grids.

Should implement
# TBD:
"""
#TBD: Should it be an AbstractArray? See notes in grid_refactor.md
# TODO: Document that grids should implement the interfaces for iteration and indexing.
abstract type Grid{T,D} end


Base.ndims(::Grid{T,D}) where {T,D} = D
Base.eltype(::Type{<:Grid{T}}) where T = T

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
    dims(grid::Grid)

Enumerate the dimensions of the grid.
"""
dims(grid::Grid) = 1:ndims(grid)
# TBD: Is this function needed? Where is it used?

# TBD: New file grid_functions.jl?
"""
TODO:

* Mention map(f,g) if you want a concrete array
"""
eval_on(g::Grid, f) = eval_on(g, f, Base.IteratorSize(g)) # TBD: Borde f vara först som i alla map, sum, och dylikt
eval_on(g::Grid, f, ::Base.HasShape) = LazyTensors.LazyFunctionArray((I...)->f(g[I...]), size(g))


"""
    getcomponent(gfun, I::Vararg{Int})

Return one of the components of gfun as a grid function.
"""
# Should it be lazy? Could it be a view?
function getcomponent(gfun, I::Vararg{Int}) end
# function getcomponent(gfun, s::Symbol) end ?
