export SecondDerivativeVariable

# """
#     SecondDerivativeVariable(grid, inner_stencil, closure_stencils, parity, direction)

# Creates a volume operator on a `Dim`-dimensional grid acting along the
# specified coordinate `direction`. The action of the operator is determined by
# the stencils `inner_stencil` and `closure_stencils`. When `Dim=1`, the
# corresponding `SecondDerivativeVariable` tensor mapping is returned. When `Dim>1`, the
# returned operator is the appropriate outer product of a one-dimensional
# operators and `IdentityMapping`s, e.g for `Dim=3` the volume operator in the
# y-direction is `I⊗op⊗I`.
# """
# function volume_operator(grid::EquidistantGrid, inner_stencil, closure_stencils, parity, direction)
#     #TODO: Check that direction <= Dim?

#     # Create 1D volume operator in along coordinate direction
#     op = SecondDerivativeVariable(restrict(grid, direction), inner_stencil, closure_stencils, parity)
#     # Create 1D IdentityMappings for each coordinate direction
#     one_d_grids = restrict.(Ref(grid), Tuple(1:dimension(grid)))
#     Is = IdentityMapping{eltype(grid)}.(size.(one_d_grids))
#     # Formulate the correct outer product sequence of the identity mappings and
#     # the volume operator
#     parts = Base.setindex(Is, op, direction)
#     return foldl(⊗, parts)
# end

"""
    SecondDerivativeVariable{T,N,M,K} <: TensorOperator{T,1}
Implements a one-dimensional constant coefficients volume operator
"""
struct SecondDerivativeVariable{T,N,M,K} <: TensorMapping{T,1,1}
    inner_stencil::NestedStencil{T,N}
    closure_stencils::NTuple{M,NestedStencil{T,K}}
    size::NTuple{1,Int}
end

function SecondDerivativeVariable(grid::EquidistantGrid{1}, inner_stencil, closure_stencils)
    return SecondDerivativeVariable(inner_stencil, Tuple(closure_stencils), size(grid))
end

closure_size(::SecondDerivativeVariable{T,N,M}) where {T,N,M} = M

LazyTensors.range_size(op::SecondDerivativeVariable) = op.size
LazyTensors.domain_size(op::SecondDerivativeVariable) = op.size

function LazyTensors.apply(op::SecondDerivativeVariable{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds apply_stencil(op.closure_stencils[Int(i)], v, Int(i))
end

function LazyTensors.apply(op::SecondDerivativeVariable{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return apply_stencil(op.inner_stencil, v, Int(i))
end

function LazyTensors.apply(op::SecondDerivativeVariable{T}, v::AbstractVector{T}, i::Index{Upper}) where T
    return @inbounds Int(op.parity)*apply_stencil_backwards(op.closure_stencils[op.size[1]-Int(i)+1], v, Int(i))
end

function LazyTensors.apply(op::SecondDerivativeVariable{T}, v::AbstractVector{T}, i) where T
    r = getregion(i, closure_size(op), op.size[1])
    return LazyTensors.apply(op, v, Index(i, r))
end
