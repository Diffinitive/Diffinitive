"""
    stencil_operator_distinct_closures(grid::EquidistantGrid, inner_stencil, lower_closure, upper_closure, direction)

Creates a multi-dimensional `StencilOperatorDistinctClosures` acting on grid functions of `grid`.

See also: [`StencilOperatorDistinctClosures`](@ref)
"""
function stencil_operator_distinct_closures(grid::EquidistantGrid, inner_stencil, lower_closure, upper_closure, direction)
    op = StencilOperatorDistinctClosures(restrict(grid, direction), inner_stencil, lower_closure, upper_closure)
    return LazyTensors.inflate(op, size(grid), direction)
end

"""
    StencilOperatorDistinctClosures{T,K,N,M,L} <: LazyTensor{T,1}

A one dimensional stencil operator with separate closures for the two boundaries.

`StencilOperatorDistinctClosures` can be contrasted to `VolumeOperator` in
that it has different closure stencils for the upper and lower boundary.
`VolumeOperator` uses the same closure for both boundaries. Having distinct
closures is useful for representing operators with skewed stencils like upwind
operators.

See also: [`VolumeOperator`](@ref), [`stencil_operator_distinct_closures`](@ref)
"""
struct StencilOperatorDistinctClosures{T,K,N,M,LC<:NTuple{N,Stencil{T,L}} where L, UC<:NTuple{M,Stencil{T,L}} where L} <: LazyTensor{T,1,1}
    inner_stencil::Stencil{T,K}
    lower_closure::LC
    upper_closure::UC
    size::Tuple{Int}
end

function StencilOperatorDistinctClosures(grid::EquidistantGrid{1}, inner_stencil, lower_closure, upper_closure)
    return StencilOperatorDistinctClosures(inner_stencil, Tuple(lower_closure), Tuple(upper_closure), size(grid))
end

lower_closure_size(::StencilOperatorDistinctClosures{T,K,N,M}) where {T,K,N,M} = N
upper_closure_size(::StencilOperatorDistinctClosures{T,K,N,M}) where {T,K,N,M} = M

LazyTensors.range_size(op::StencilOperatorDistinctClosures) = op.size
LazyTensors.domain_size(op::StencilOperatorDistinctClosures) = op.size

function LazyTensors.apply(op::StencilOperatorDistinctClosures, v::AbstractVector, i::Index{Lower})
    return @inbounds apply_stencil(op.lower_closure[Int(i)], v, Int(i))
end

function LazyTensors.apply(op::StencilOperatorDistinctClosures, v::AbstractVector, i::Index{Interior})
    return apply_stencil(op.inner_stencil, v, Int(i))
end

function LazyTensors.apply(op::StencilOperatorDistinctClosures, v::AbstractVector, i::Index{Upper})
    stencil_index = Int(i) - (op.size[1]-upper_closure_size(op))
    return @inbounds apply_stencil(op.upper_closure[stencil_index], v, Int(i))
end

function LazyTensors.apply(op::StencilOperatorDistinctClosures, v::AbstractVector, i)
    if i <= lower_closure_size(op)
        LazyTensors.apply(op, v, Index(i, Lower))
    elseif i > op.size[1]-upper_closure_size(op)
        LazyTensors.apply(op, v, Index(i, Upper))
    else
        LazyTensors.apply(op, v, Index(i, Interior))
    end
end
# TODO: Move this to LazyTensors when we have the region communication down.
