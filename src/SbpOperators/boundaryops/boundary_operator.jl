"""
    BoundaryOperator{B} <: LazyTensor{0,1}

Implements the boundary operator `op` for 1D as a `LazyTensor`

`op` is the restriction of a grid function to the boundary using some closure
stencil `ST<:Stencil`. The boundary to restrict to is determined by `B`. `op'` is the
prolongation of a zero dimensional array to the whole grid using the same
closure stencil.
"""
struct BoundaryOperator{B<:BoundaryIdentifier,ST<:Stencil} <: LazyTensor{0,1}
    stencil::ST
    size::Int
end

"""
    BoundaryOperator(grid::EquidistantGrid, closure_stencil, boundary)

Constructs the BoundaryOperator with stencil `closure_stencil` for a
`EquidistantGrid` `grid`, restricting to to the boundary specified by
`boundary`.
"""
function BoundaryOperator(grid::EquidistantGrid, closure_stencil::Stencil, boundary::BoundaryIdentifier)
    B = typeof(boundary)
    ST = typeof(closure_stencil)
    return BoundaryOperator{B,ST}(closure_stencil,size(grid)[1])
end

"""
    closure_size(::BoundaryOperator)

The size of the closure stencil.
"""
closure_size(op::BoundaryOperator) = length(op.stencil)

LazyTensors.range_size(op::BoundaryOperator) = ()
LazyTensors.domain_size(op::BoundaryOperator) = (op.size,)

function Base.adjoint(op::BoundaryOperator{B,ST}) where {B,ST}
    return BoundaryOperatorAdjoint{B,ST}(op.stencil, op.size)
end

function LazyTensors.apply(op::BoundaryOperator{LowerBoundary}, v::AbstractVector)
    apply_stencil(op.stencil,v,1)
end

function LazyTensors.apply(op::BoundaryOperator{UpperBoundary}, v::AbstractVector)
    apply_stencil_backwards(op.stencil,v,op.size)
end

# REVIEW: An alternative would be if we could take the adjoint of a Stencil to get a
# a new stencil. We would still need the BoundaryOperatorAdjoint type since the adjoint
# operator has swapped dimensions for range and dim, but the implementation would otherwise
# look the same as for `BoundaryOperator`?
# The same approach would also be useful for constructing adjoint operators of other 
# base 1D operators, e.g. a `VolumeOperator`.
"""
    BoundaryOperatorAdjoint{B} <: LazyTensor{1,0}

Implements the adjoint of a `BoundaryOperator`.
"""
struct BoundaryOperatorAdjoint{B<:BoundaryIdentifier,ST<:Stencil} <: LazyTensor{1,0}
    stencil::ST
    size::Int
end

closure_size(op::BoundaryOperatorAdjoint) = length(op.stencil)

LazyTensors.range_size(op::BoundaryOperatorAdjoint) = (op.size,)
LazyTensors.domain_size(op::BoundaryOperatorAdjoint) = ()

function Base.adjoint(op::BoundaryOperatorAdjoint{B,ST}) where {B,ST}
    return BoundaryOperator{B,ST}(op.stencil, op.size)
end

function LazyTensors.apply(op::BoundaryOperatorAdjoint{LowerBoundary}, v::AbstractArray{<:Any,0}, i::Index{Lower})
    return op.stencil[Int(i)-1]*v[]
end

function LazyTensors.apply(op::BoundaryOperatorAdjoint{UpperBoundary}, v::AbstractArray{<:Any,0}, i::Index{Upper})
    return op.stencil[op.size - Int(i)]*v[]
end

# Catch all combinations of Lower, Upper and Interior not caught by the two previous methods.
function LazyTensors.apply(op::BoundaryOperatorAdjoint, v::AbstractArray{<:Any,0}, i::Index)
    return zero(eltype(v))
end

function LazyTensors.apply(op::BoundaryOperatorAdjoint, v::AbstractArray{<:Any,0}, i)
    r = getregion(i, closure_size(op), op.size)
    apply(op, v, Index(i,r))
end
