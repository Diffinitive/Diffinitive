"""
     Grid{T,D,RD} <: AbstractArray{T,D}

The top level type for grids.

Should implement
# TBD:
"""
#TBD: Should it be an AbstractArray? See notes in grid_refactor.md
abstract type Grid{T,D,RD} end


Base.ndims(::Grid{T,D,RD}) where {T,D,RD} = D # nidms borde nog vara antalet index som används för att indexera nätet. Snarare än vilken dimension nätet har (tänk ostrukturerat)
nrangedims(::Grid{T,D,RD}) where {T,D,RD} = RD
Base.eltype(::Grid{T,D,RD}) where {T,D,RD} = T # vad ska eltype vara? Inte T väl... en vektor? SVector{T,D}?

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
