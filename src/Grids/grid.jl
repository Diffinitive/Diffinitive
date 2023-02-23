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

function refine(::Grid) end
function coarsen(::Grid) end # Should this be here? What if it is not possible?

"""
# TODO
"""
function boundary_identifiers end
"""
# TODO
"""
function boundary_grid end


# TODO: Make sure that all grids implement all of the above.

"""
    dims(grid::Grid)

Enumerate the dimensions of the grid.
"""
dims(grid::Grid) = 1:ndims(grid)


# TBD: New file grid_functions.jl?

function eval_on(::Grid) end # TODO: Should return a LazyArray and index the grid

"""
    getcomponent(gfun, I::Vararg{Int})

Return one of the components of gfun as a grid function.
"""
# Should it be lazy? Could it be a view?
function getcomponent(gfun, I::Vararg{Int}) end
# function getcomponent(gfun, s::Symbol) end ?
