"""
    BoundaryOperator{T,B,N} <: LazyTensor{T,0,1}

Implements the boundary operator `op` for 1D as a `LazyTensor`

`op` is the restriction of a grid function to the boundary using some closure
`Stencil{T,N}`. The boundary to restrict to is determined by `B`. `op'` is the
prolongation of a zero dimensional array to the whole grid using the same
closure stencil.
"""
struct BoundaryOperator{T,B<:BoundaryIdentifier,N} <: LazyTensor{T,0,1}
    stencil::Stencil{T,N}
    size::Int
end

"""
    BoundaryOperator(grid::EquidistantGrid, closure_stencil, boundary)

Constructs the BoundaryOperator with stencil `closure_stencil` for a
`EquidistantGrid` `grid`, restricting to to the boundary specified by
`boundary`.
"""
function BoundaryOperator(grid::EquidistantGrid, closure_stencil::Stencil{T,N}, boundary::BoundaryIdentifier) where {T,N}
    return BoundaryOperator{T,typeof(boundary),N}(closure_stencil,size(grid)[1])
end

"""
    closure_size(::BoundaryOperator)

The size of the closure stencil.
"""
closure_size(::BoundaryOperator{T,B,N}) where {T,B,N} = N

LazyTensors.range_size(op::BoundaryOperator) = ()
LazyTensors.domain_size(op::BoundaryOperator) = (op.size,)

function LazyTensors.apply(op::BoundaryOperator{<:Any,LowerBoundary}, v::AbstractVector)
    apply_stencil(op.stencil,v,1)
end

function LazyTensors.apply(op::BoundaryOperator{<:Any,UpperBoundary}, v::AbstractVector)
    apply_stencil_backwards(op.stencil,v,op.size)
end

function LazyTensors.apply_transpose(op::BoundaryOperator{<:Any,LowerBoundary}, v::AbstractArray{<:Any,0}, i::Index{Lower})
    return op.stencil[Int(i)-1]*v[]
end

function LazyTensors.apply_transpose(op::BoundaryOperator{<:Any,UpperBoundary}, v::AbstractArray{<:Any,0}, i::Index{Upper})
    return op.stencil[op.size[1] - Int(i)]*v[]
end

# Catch all combinations of Lower, Upper and Interior not caught by the two previous methods.
function LazyTensors.apply_transpose(op::BoundaryOperator, v::AbstractArray{<:Any,0}, i::Index)
    return zero(eltype(v))
end

function LazyTensors.apply_transpose(op::BoundaryOperator, v::AbstractArray{<:Any,0}, i)
    r = getregion(i, closure_size(op), op.size)
    apply_transpose(op, v, Index(i,r))
end
