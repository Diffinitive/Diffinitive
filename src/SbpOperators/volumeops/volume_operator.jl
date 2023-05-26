"""
    VolumeOperator{T,N,M,K} <: LazyTensor{T,1,1}

A one-dimensional constant coefficients stencil operator.
"""
struct VolumeOperator{T,N,M,K} <: LazyTensor{T,1,1}
    inner_stencil::Stencil{T,N}
    closure_stencils::NTuple{M,Stencil{T,K}}
    size::NTuple{1,Int}
    parity::Parity
end

function VolumeOperator(grid::EquidistantGrid, inner_stencil, closure_stencils, parity)
    return VolumeOperator(inner_stencil, Tuple(closure_stencils), size(grid), parity)
end # TBD: Remove this function?

closure_size(::VolumeOperator{T,N,M}) where {T,N,M} = M

LazyTensors.range_size(op::VolumeOperator) = op.size
LazyTensors.domain_size(op::VolumeOperator) = op.size

function LazyTensors.apply(op::VolumeOperator, v::AbstractVector, i::Index{Lower})
    return @inbounds apply_stencil(op.closure_stencils[Int(i)], v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator, v::AbstractVector, i::Index{Interior})
    return apply_stencil(op.inner_stencil, v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator, v::AbstractVector, i::Index{Upper})
    return @inbounds Int(op.parity)*apply_stencil_backwards(op.closure_stencils[op.size[1]-Int(i)+1], v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator, v::AbstractVector, i)
    r = getregion(i, closure_size(op), op.size[1])
    return LazyTensors.apply(op, v, Index(i, r))
end
# TODO: Move this to LazyTensors when we have the region communication down.
