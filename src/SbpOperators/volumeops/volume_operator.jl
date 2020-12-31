"""
    volume_operator(grid,inner_stencil,closure_stencils,parity,direction)
Creates a volume operator on a `Dim`-dimensional grid acting along the
specified coordinate `direction`. The action of the operator is determined by the
stencils `inner_stencil` and `closure_stencils`.
When `Dim=1`, the corresponding `VolumeOperator` tensor mapping is returned.
When `Dim>1`, the `VolumeOperator` `op` is inflated by the outer product
of `IdentityMappings` in orthogonal coordinate directions, e.g for `Dim=3`,
the boundary restriction operator in the y-direction direction is `Ix⊗op⊗Iz`.
"""
function volume_operator(grid::EquidistantGrid{Dim,T}, inner_stencil::Stencil{T}, closure_stencils::NTuple{M,Stencil{T}}, parity, direction) where {Dim,T,M}
    #TODO: Check that direction <= Dim?

    # Create 1D volume operator in along coordinate direction
    op = VolumeOperator(restrict(grid, direction), inner_stencil, closure_stencils, parity)
    # Create 1D IdentityMappings for each coordinate direction
    one_d_grids = restrict.(Ref(grid), Tuple(1:Dim))
    Is = IdentityMapping{T}.(size.(one_d_grids))
    # Formulate the correct outer product sequence of the identity mappings and
    # the volume operator
    parts = Base.setindex(Is, op, direction)
    return foldl(⊗, parts)
end

"""
    VolumeOperator{T,N,M,K} <: TensorOperator{T,1}
Implements a one-dimensional constant coefficients volume operator
"""
struct VolumeOperator{T,N,M,K} <: TensorMapping{T,1,1}
    inner_stencil::Stencil{T,N}
    closure_stencils::NTuple{M,Stencil{T,K}}
    size::NTuple{1,Int}
    parity::Parity
end

function VolumeOperator(grid::EquidistantGrid{1}, inner_stencil, closure_stencils, parity)
    return VolumeOperator(inner_stencil, closure_stencils, size(grid), parity)
end

closure_size(::VolumeOperator{T,N,M}) where {T,N,M} = M

LazyTensors.range_size(op::VolumeOperator) = op.size
LazyTensors.domain_size(op::VolumeOperator) = op.size

function LazyTensors.apply(op::VolumeOperator{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds apply_stencil(op.closure_stencils[Int(i)], v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return apply_stencil(op.inner_stencil, v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator{T}, v::AbstractVector{T}, i::Index{Upper}) where T
    return @inbounds Int(op.parity)*apply_stencil_backwards(op.closure_stencils[op.size[1]-Int(i)+1], v, Int(i))
end

function LazyTensors.apply(op::VolumeOperator{T}, v::AbstractVector{T}, i) where T
    r = getregion(i, closure_size(op), op.size[1])
    return LazyTensors.apply(op, v, Index(i, r))
end
