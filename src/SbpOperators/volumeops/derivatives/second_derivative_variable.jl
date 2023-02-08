export SecondDerivativeVariable

# REVIEW: Fixa docs
"""
    SecondDerivativeVariable{Dir,T,D,...} <: LazyTensor{T,D,D}

A second derivative operator in direction `Dir` with a variable coefficient.
"""
struct SecondDerivativeVariable{Dir,T,D,M,IStencil<:NestedStencil{T},CStencil<:NestedStencil{T},TArray<:AbstractArray} <: LazyTensor{T,D,D}
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
    check_coefficient(grid, coeff)

    Δxᵢ = spacing(grid)[dir]
    scaled_inner_stencil = scale(inner_stencil, 1/Δxᵢ^2)
    scaled_closure_stencils = scale.(Tuple(closure_stencils), 1/Δxᵢ^2)
    return SecondDerivativeVariable{dir, ndims(grid)}(scaled_inner_stencil, scaled_closure_stencils, size(grid), coeff)
end

function SecondDerivativeVariable(grid::EquidistantGrid{1}, coeff::AbstractVector, inner_stencil::NestedStencil, closure_stencils)
    return SecondDerivativeVariable(grid, coeff, inner_stencil, closure_stencils, 1)
end

@doc raw"""
    SecondDerivativeVariable(grid::EquidistantGrid, coeff::AbstractArray, stencil_set, dir)

Create a `LazyTensor` for the second derivative with a variable coefficient
`coeff` on `grid` from the stencils in `stencil_set`. The direction is
determined by `dir`.

`coeff` is a grid function on `grid`.

# Example
With
```
D = SecondDerivativeVariable(g, c, stencil_set, 2)
```
then `D*u` approximates
```math
\frac{\partial}{\partial y} c(x,y) \frac{\partial u}{\partial y},
```
on ``(0,1)⨯(0,1)`` represented by `g`.
"""
function SecondDerivativeVariable(grid::EquidistantGrid, coeff::AbstractArray, stencil_set, dir::Int)
    inner_stencil    = parse_nested_stencil(eltype(coeff), stencil_set["D2variable"]["inner_stencil"])
    closure_stencils = parse_nested_stencil.(eltype(coeff), stencil_set["D2variable"]["closure_stencils"])

    return SecondDerivativeVariable(grid, coeff, inner_stencil, closure_stencils, dir)
end

function check_coefficient(grid, coeff)
    if ndims(grid) != ndims(coeff)
        throw(ArgumentError("The coefficient has dimension $(ndims(coeff)) while the grid is dimension $(ndims(grid))"))
    end

    if size(grid) != size(coeff)
        throw(DimensionMismatch("the size $(size(coeff)) of the coefficient does not match the size $(size(grid)) of the grid"))
    end
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
    elseif I[derivative_direction(op)] isa Index{Interior}
        return apply_interior(op, v, Int.(I)...)
    else
        error("Invalid region")
    end
end

function LazyTensors.apply(op::SecondDerivativeVariable, v::AbstractArray, I...)
    dir = derivative_direction(op)

    i = I[dir]

    I = map(i->Index(i, Interior), I)
    if 0 < i <= closure_size(op)
        I = Base.setindex(I, Index(i, Lower), dir)
        return LazyTensors.apply(op, v, I...)
    elseif closure_size(op) < i <= op.size[dir]-closure_size(op)
        I = Base.setindex(I, Index(i, Interior), dir)
        return LazyTensors.apply(op, v, I...)
    elseif op.size[dir]-closure_size(op) < i <= op.size[dir]
        I = Base.setindex(I, Index(i, Upper), dir)
        return LazyTensors.apply(op, v, I...)
    else
        error("Bounds error") # TODO: Make this more standard
    end
end


## 2D Specific implementations to avoid instability
## TODO: Should really be solved by fixing the general methods instead


## x-direction
function apply_lower(op::SecondDerivativeVariable{1}, v, i, j)
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    return @inbounds apply_stencil(op.closure_stencils[i], c̃, ṽ, i)
end

function apply_interior(op::SecondDerivativeVariable{1}, v, i, j)
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    return @inbounds apply_stencil(op.inner_stencil, c̃, ṽ, i)
end

function apply_upper(op::SecondDerivativeVariable{1}, v, i, j)
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    stencil = op.closure_stencils[op.size[derivative_direction(op)]-i+1]
    return @inbounds apply_stencil_backwards(stencil, c̃, ṽ, i)
end


## y-direction
function apply_lower(op::SecondDerivativeVariable{2}, v, i, j)
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    return @inbounds apply_stencil(op.closure_stencils[j], c̃, ṽ, j)
end

function apply_interior(op::SecondDerivativeVariable{2}, v, i, j)
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    return @inbounds apply_stencil(op.inner_stencil, c̃, ṽ, j)
end

function apply_upper(op::SecondDerivativeVariable{2}, v, i, j)
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    stencil = op.closure_stencils[op.size[derivative_direction(op)]-j+1]
    return @inbounds apply_stencil_backwards(stencil, c̃, ṽ, j)
end
