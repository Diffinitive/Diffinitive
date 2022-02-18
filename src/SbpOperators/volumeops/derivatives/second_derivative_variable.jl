export SecondDerivativeVariable

# REVIEW: Fixa docs
"""
    SecondDerivativeVariable{Dir,T,D,...} <: TensorMapping{T,D,D}

A second derivative operator in direction `Dir` with a variable coefficient.
"""
struct SecondDerivativeVariable{Dir,T,D,M,IStencil<:NestedStencil{T},CStencil<:NestedStencil{T},TArray<:AbstractArray} <: TensorMapping{T,D,D}
    inner_stencil::IStencil
    closure_stencils::NTuple{M,CStencil}
    size::NTuple{D,Int}
    coefficient::TArray

    function SecondDerivativeVariable{Dir, D}(inner_stencil::NestedStencil{T}, closure_stencils::NTuple{M,NestedStencil{T}}, size::NTuple{D,Int}, coefficient::AbstractArray) where {Dir,T,D,M}
        IStencil = typeof(inner_stencil)
        CStencil = eltype(closure_stencils)
        TArray = typeof(coefficient)
        return new{Dir,T,D,M,IStencil,CStencil,TArray}(inner_stencil,closure_stencils,size, coefficient)
    end
end

function SecondDerivativeVariable(grid::EquidistantGrid, coeff::AbstractArray, inner_stencil, closure_stencils, dir)
    Δxᵢ = spacing(grid)[dir]
    scaled_inner_stencil = scale(inner_stencil, 1/Δxᵢ^2)
    scaled_closure_stencils = scale.(Tuple(closure_stencils), 1/Δxᵢ^2)
    return SecondDerivativeVariable{dir, dimension(grid)}(scaled_inner_stencil, scaled_closure_stencils, size(grid), coeff)
end

function SecondDerivativeVariable(grid::EquidistantGrid{1}, coeff::AbstractVector, inner_stencil, closure_stencils)
    return SecondDerivativeVariable(grid, coeff, inner_stencil, closure_stencils, 1)
end

function SecondDerivativeVariable(grid::EquidistantGrid, coeff::AbstractArray, stencil_set, dir)
    inner_stencil    = parse_nested_stencil(eltype(coeff), stencil_set["D2variable"]["inner_stencil"])
    closure_stencils = parse_nested_stencil.(eltype(coeff), stencil_set["D2variable"]["closure_stencils"])

    return SecondDerivativeVariable(grid, coeff, inner_stencil, closure_stencils, dir)
end

derivative_direction(::SecondDerivativeVariable{Dir}) where {Dir} = Dir

closure_size(op::SecondDerivativeVariable) = length(op.closure_stencils)

LazyTensors.range_size(op::SecondDerivativeVariable) = op.size
LazyTensors.domain_size(op::SecondDerivativeVariable) = op.size


function derivative_view(op, a, I)
    d = derivative_direction(op)

    Iview = Base.setindex(I,:,d)
    return @view a[Iview...]
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
    stencil = op.closure_stencils[op.size[derivative_direction(op)]-i+1]
    return @inbounds apply_stencil_backwards(stencil, c̃, ṽ, i)
end

function LazyTensors.apply(op::SecondDerivativeVariable, v::AbstractArray, I::Vararg{Index})
    if I[derivative_direction(op)] isa Index{Lower}
        return apply_lower(op, v, Int.(I)...)
    elseif I[derivative_direction(op)] isa Index{Upper}
        return apply_upper(op, v, Int.(I)...)
    else
        return apply_interior(op, v, Int.(I)...)
    end
end

function LazyTensors.apply(op::SecondDerivativeVariable, v::AbstractArray, I...)
    dir = derivative_direction(op)

    i = I[dir]
    r = getregion(i, closure_size(op), op.size[dir])

    I = map(i->Index(i, Interior), I)
    I = Base.setindex(I, Index(i, r), dir)
    return LazyTensors.apply(op, v, I...)
end
