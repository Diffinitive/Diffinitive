"""
    MultiBlockBoundary{N, BID} <: BoundaryIdentifier

A boundary identifier for a multiblock grids. `N` Specifies which grid and
`BID` which boundary on that grid.
"""
struct MultiBlockBoundary{N, BID} <: BoundaryIdentifier end
grid_id(::MultiBlockBoundary{N, BID}) where {N, BID} = N
boundary_id(::MultiBlockBoundary{N, BID}) where {N, BID} = BID()
