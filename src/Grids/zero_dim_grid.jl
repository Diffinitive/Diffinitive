"""
    ZeroDimGrid{T} <: Grid{T,0}

A zero dimensional grid consisting of a single point.
"""
struct ZeroDimGrid{T} <: Grid{T,0}
    point::T
end

# Indexing interface
Base.getindex(g::ZeroDimGrid) = g.point
Base.eachindex(g::ZeroDimGrid) = CartesianIndices(())

# Iteration interface
Base.iterate(g::ZeroDimGrid) = (g.point, nothing)
Base.iterate(g::ZeroDimGrid, ::Any) = nothing

Base.IteratorSize(::Type{<:ZeroDimGrid}) = Base.HasShape{0}()
Base.length(g::ZeroDimGrid) = 1
Base.size(g::ZeroDimGrid) = ()


refine(g::ZeroDimGrid, ::Int) = g
coarsen(g::ZeroDimGrid, ::Int) = g

boundary_identifiers(g::ZeroDimGrid) = ()
boundary_grid(g::ZeroDimGrid, ::Any) = throw(ArgumentError("ZeroDimGrid has no boundaries"))
boundary_indices(g::ZeroDimGrid, ::Any) = throw(ArgumentError("ZeroDimGrid has no boundaries"))
