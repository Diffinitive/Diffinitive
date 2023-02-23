"""
    ZeroDimGrid{T} <: Grid{T,0}
# TODO
"""
struct ZeroDimGrid{T} <: Grid{T,0}
    point::T
end

# # Indexing interface
Base.getindex(g::ZeroDimGrid) = g.point
Base.eachindex(g::ZeroDimGrid) = CartesianIndices(())

# # Iteration interface
Base.iterate(g::ZeroDimGrid) = (g.point, nothing)
Base.iterate(g::ZeroDimGrid, ::Any) = nothing

Base.IteratorSize(::Type{ZeroDimGrid{T}}) where T = Base.HasShape{0}()
Base.eltype(::Type{ZeroDimGrid{T}}) where T = T
Base.length(g::ZeroDimGrid) = 1
Base.size(g::ZeroDimGrid) = ()
