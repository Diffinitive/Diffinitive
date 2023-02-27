"""
# TODO:
"""
#TODO: Document recomendations for type of range. (LinRange is faster?)
struct EquidistantGrid{T,R<:AbstractRange{T}} <: Grid{T,1}
    points::R
end

Base.eachindex(g::EquidistantGrid) = eachindex(g.points)

# Indexing interface
Base.getindex(g::EquidistantGrid, i) = g.points[i]
Base.firstindex(g::EquidistantGrid) = firstindex(g.points)
Base.lastindex(g::EquidistantGrid) = lastindex(g.points)

# Iteration interface
Base.iterate(g::EquidistantGrid) = iterate(g.points)
Base.iterate(g::EquidistantGrid, state) = iterate(g.points, state)

Base.IteratorSize(::Type{<:EquidistantGrid}) = Base.HasShape{1}()
Base.length(g::EquidistantGrid) = length(g.points)
Base.size(g::EquidistantGrid) = size(g.points)


"""
    spacing(grid::EquidistantGrid)

The spacing between grid points.
"""
spacing(g::EquidistantGrid) = step(g.points)


"""
    inverse_spacing(grid::EquidistantGrid)

The reciprocal of the spacing between grid points.
"""
inverse_spacing(g::EquidistantGrid) = 1/step(g.points)


boundary_identifiers(::EquidistantGrid) = (Lower(), Upper())
boundary_grid(g::EquidistantGrid, id::Lower) = ZeroDimGrid(g[begin])
boundary_grid(g::EquidistantGrid, id::Upper) = ZeroDimGrid(g[end])


"""
    refine(g::EquidistantGrid, r::Int)

Refines `grid` by a factor `r`. The factor is applied to the number of
intervals which is 1 less than the size of the grid.

See also: [`coarsen`](@ref)
"""
function refine(g::EquidistantGrid, r::Int)
    new_sz = (length(g) - 1)*r + 1
    return EquidistantGrid(change_length(g.points, new_sz))
end

"""
    coarsen(grid::EquidistantGrid, r::Int)

Coarsens `grid` by a factor `r`. The factor is applied to the number of
intervals which is 1 less than the size of the grid. If the number of
intervals are not divisible by `r` an error is raised.

See also: [`refine`](@ref)
"""
function coarsen(g::EquidistantGrid, r::Int)
    if (length(g)-1)%r != 0
        throw(DomainError(r, "Size minus 1 must be divisible by the ratio."))
    end

    new_sz = (length(g) - 1)÷r + 1

    return EquidistantGrid(change_length(g.points, new_sz))
end


"""
    equidistant_grid(size::Dims, limit_lower, limit_upper)

Construct an equidistant grid with corners at the coordinates `limit_lower` and
`limit_upper`.

The length of the domain sides are given by the components of
`limit_upper-limit_lower`. E.g for a 2D grid with `limit_lower=(-1,0)` and
`limit_upper=(1,2)` the domain is defined as `(-1,1)x(0,2)`. The side lengths
of the grid are not allowed to be negative.

The number of equidistantly spaced points in each coordinate direction are given
by the tuple `size`.

Note: If `limit_lower` and `limit_upper` are integers and `size` would allow a
completely integer grid, `equidistant_grid` will still return a floating point
grid. This simlifies the implementation and avoids certain surprise
behaviours.
"""
function equidistant_grid(size::Dims, limit_lower, limit_upper)
    if any(size .<= 0)
        throw(DomainError("all components of size must be postive"))
    end

    if any(limit_upper.-limit_lower .<= 0)
        throw(DomainError("all side lengths must be postive"))
    end

    gs = map(size, limit_lower, limit_upper) do s,l,u
        EquidistantGrid(range(l, u, length=s)) # TBD: Should it use LinRange instead?
    end

    return TensorGrid(gs...)
end


"""
    equidistant_grid(size::Int, limit_lower::T, limit_upper::T)

Constructs a 1D equidistant grid.
"""
function equidistant_grid(size::Int, limit_lower::T, limit_upper::T) where T
    # TBD: Should this really return a TensorGrid?
	return equidistant_grid((size,),(limit_lower,),(limit_upper,))
end

CartesianBoundary{D,BID} = TensorGridBoundary{D,BID} # TBD: What should we do about the naming of this boundary?


"""
    change_length(::AbstractRange, n)

Change the length of a range to `n`, keeping the same start and stop.
"""
function change_length end

change_length(r::UnitRange, n) = StepRange{Int,Int}(range(r[begin], r[end], n))
change_length(r::StepRange, n) = StepRange{Int,Int}(range(r[begin], r[end], n))
change_length(r::StepRangeLen, n) = range(r[begin], r[end], n)
change_length(r::LinRange, n) = LinRange(r[begin], r[end], n)
