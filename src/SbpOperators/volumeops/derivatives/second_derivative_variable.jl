"""
    second_derivative_variable(g, coeff ..., [direction])

The variable second derivative operator as a `LazyTensor` on the given grid.
`coeff` is a grid function of the variable coefficient.

Approximates the d/dξ c d/dξ on `g` along the coordinate dimension specified
by `direction`.
"""
function second_derivative_variable end

function second_derivative_variable(g::TensorGrid, coeff, stencil_set, dir::Int)
    inner_stencil    = parse_nested_stencil(eltype(coeff), stencil_set["D2variable"]["inner_stencil"])
    closure_stencils = parse_nested_stencil.(eltype(coeff), stencil_set["D2variable"]["closure_stencils"])

    return second_derivative_variable(g, coeff, inner_stencil, closure_stencils, dir)
end

function second_derivative_variable(g::EquidistantGrid, coeff, stencil_set)
    return second_derivative_variable(TensorGrid(g), coeff, stencil_set, 1)
end

function second_derivative_variable(g::TensorGrid, coeff, inner_stencil::NestedStencil, closure_stencils, dir)
    check_coefficient(g, coeff)

    Δxᵢ = spacing(g.grids[dir])
    scaled_inner_stencil = scale(inner_stencil, 1/Δxᵢ^2)
    scaled_closure_stencils = scale.(Tuple(closure_stencils), 1/Δxᵢ^2)
    return SecondDerivativeVariable(coeff, scaled_inner_stencil, scaled_closure_stencils, dir)
end

function check_coefficient(g, coeff)
    if ndims(g) != ndims(coeff)
        throw(ArgumentError("The coefficient has dimension $(ndims(coeff)) while the grid is dimension $(ndims(g))"))
    end

    if size(g) != size(coeff)
        throw(DimensionMismatch("the size $(size(coeff)) of the coefficient does not match the size $(size(g)) of the grid"))
    end
end


"""
    SecondDerivativeVariable{Dir,T,D,...} <: LazyTensor{T,D,D}

A second derivative operator in direction `Dir` with a variable coefficient.
"""
struct SecondDerivativeVariable{Dir,T,D,M,IStencil<:NestedStencil{T},CStencil<:NestedStencil{T},TArray<:AbstractArray} <: LazyTensor{T,D,D}
    inner_stencil::IStencil
    closure_stencils::NTuple{M,CStencil}
    coefficient::TArray

    function SecondDerivativeVariable(coefficient::AbstractArray, inner_stencil::NestedStencil{T}, closure_stencils::NTuple{M,NestedStencil{T}}, dir) where {T,M}
        D = ndims(coefficient)
        IStencil = typeof(inner_stencil)
        CStencil = eltype(closure_stencils)
        TArray = typeof(coefficient)
        return new{dir,T,D,M,IStencil,CStencil,TArray}(inner_stencil, closure_stencils, coefficient)
    end
end

derivative_direction(::SecondDerivativeVariable{Dir}) where {Dir} = Dir

closure_size(op::SecondDerivativeVariable) = length(op.closure_stencils)

LazyTensors.range_size(op::SecondDerivativeVariable) = size(op.coefficient)
LazyTensors.domain_size(op::SecondDerivativeVariable) = size(op.coefficient)


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
    sz = domain_size(op)[derivative_direction(op)]
    stencil = op.closure_stencils[sz-i+1]
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
    sz = domain_size(op)[dir]

    i = I[dir]

    I = map(i->Index(i, Interior), I)
    if 0 < i <= closure_size(op)
        I = Base.setindex(I, Index(i, Lower), dir)
        return LazyTensors.apply(op, v, I...)
    elseif closure_size(op) < i <= sz-closure_size(op)
        I = Base.setindex(I, Index(i, Interior), dir)
        return LazyTensors.apply(op, v, I...)
    elseif sz-closure_size(op) < i <= sz
        I = Base.setindex(I, Index(i, Upper), dir)
        return LazyTensors.apply(op, v, I...)
    else
        error("Bounds error") # This should be `throw(BoundsError())` but the type inference is so fragile that it doesn't work. Needs investigation. / Jonatan 2023-06-08
    end
end


# 2D Specific implementations to avoid type instability
# TBD: Can this be solved by fixing the general methods instead?


## x-direction
function apply_lower(op::SecondDerivativeVariable{1}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    return @inbounds apply_stencil(op.closure_stencils[i], c̃, ṽ, i)
end

function apply_interior(op::SecondDerivativeVariable{1}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    return @inbounds apply_stencil(op.inner_stencil, c̃, ṽ, i)
end

function apply_upper(op::SecondDerivativeVariable{1}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[:,j]
    c̃ = @view op.coefficient[:,j]

    sz = domain_size(op)[derivative_direction(op)]
    stencil = op.closure_stencils[sz-i+1]
    return @inbounds apply_stencil_backwards(stencil, c̃, ṽ, i)
end


## y-direction
function apply_lower(op::SecondDerivativeVariable{2}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    return @inbounds apply_stencil(op.closure_stencils[j], c̃, ṽ, j)
end

function apply_interior(op::SecondDerivativeVariable{2}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    return @inbounds apply_stencil(op.inner_stencil, c̃, ṽ, j)
end

function apply_upper(op::SecondDerivativeVariable{2}, v, i, j)
    Base.@constprop :aggressive
    ṽ = @view v[i,:]
    c̃ = @view op.coefficient[i,:]

    sz = domain_size(op)[derivative_direction(op)]
    stencil = op.closure_stencils[sz-j+1]
    return @inbounds apply_stencil_backwards(stencil, c̃, ṽ, j)
end
