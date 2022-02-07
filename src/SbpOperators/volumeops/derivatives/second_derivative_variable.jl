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

Implements the one-dimensional second derivative with variable coefficients.
"""
struct SecondDerivativeVariable{Dir,T,D,N,M,K,TArray<:AbstractArray} <: TensorMapping{T,D,D}
    inner_stencil::NestedStencil{T,N}
    closure_stencils::NTuple{M,NestedStencil{T,K}}
    size::NTuple{D,Int}
    coefficient::TArray

    function SecondDerivativeVariable{Dir, D}(inner_stencil::NestedStencil{T,N}, closure_stencils::NTuple{M,NestedStencil{T,K}}, size::NTuple{D,Int}, coefficient::TArray) where {Dir,T,D,N,M,K,TArray<:AbstractArray}
        return new{Dir,T,D,N,M,K,TArray}(inner_stencil,closure_stencils,size, coefficient)
    end
end

function SecondDerivativeVariable(grid::EquidistantGrid, coeff::AbstractArray, inner_stencil, closure_stencils, dir)
    return SecondDerivativeVariable{dir, dimension(grid)}(inner_stencil, Tuple(closure_stencils), size(grid), coeff)
end

function SecondDerivativeVariable(grid::EquidistantGrid{1}, coeff::AbstractVector, inner_stencil, closure_stencils)
    return SecondDerivativeVariable(grid, coeff, inner_stencil, closure_stencils, 1)
end

derivative_direction(::SecondDerivativeVariable{Dir}) where {Dir} = Dir

closure_size(::SecondDerivativeVariable{T,N,M}) where {T,N,M} = M

LazyTensors.range_size(op::SecondDerivativeVariable) = op.size
LazyTensors.domain_size(op::SecondDerivativeVariable) = op.size


function derivative_view(op, a, I)
    d = derivative_direction(op)

    Iview = Base.setindex(I,:,d)
    return @view a[Iview...]

    # D = domain_dim(op)
    # Iₗ, _, Iᵣ = split_tuple(I, Val(d-1), Val(1),  Val(D-d))
    # return @view a[Iₗ..., :, Iᵣ...]
end

function apply_lower(op::SecondDerivativeVariable, v, I...)
    ṽ = derivative_view(op, v, I)
    c̃ = derivative_view(op, op.coefficient, I)

    i = I[derivative_direction(op)]
    return @inbounds apply_stencil(op.closure_stencils[i], c̃, ṽ, i)
end

function apply_interior(op::SecondDerivativeVariable, v, I...)
    ṽ = derivative_view(op, v, I)
    c̃ = derivative_view(op, op.coefficient, I)

    i = I[derivative_direction(op)]
    return apply_stencil(op.inner_stencil, c̃, ṽ, i)
end

function apply_upper(op::SecondDerivativeVariable, v, I...)
    ṽ = derivative_view(op, v, I)
    c̃ = derivative_view(op, op.coefficient, I)

    i = I[derivative_direction(op)]
    return @inbounds apply_stencil_backwards(op.closure_stencils[op.size[1]-i+1], c̃, ṽ, i)
end

function LazyTensors.apply(op::SecondDerivativeVariable, v::AbstractVector, I::Vararg{Index})
    if I[derivative_direction(op)] isa Index{Lower}
        return apply_lower(op, v, Int.(I)...)
    elseif I[derivative_direction(op)] isa Index{Upper}
        return apply_upper(op, v, Int.(I)...)
    else
        return apply_interior(op, v, Int.(I)...)
    end
end

function LazyTensors.apply(op::SecondDerivativeVariable, v::AbstractVector, I...)
    i = I[derivative_direction(op)]
    r = getregion(i, closure_size(op), op.size[1])
    return LazyTensors.apply(op, v, Index(i, r))
end

# TODO: Rename SecondDerivativeVariable -> VariableSecondDerivative
