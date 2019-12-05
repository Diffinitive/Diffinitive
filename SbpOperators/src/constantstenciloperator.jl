abstract type ConstantStencilOperator end

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function apply_2nd_derivative(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, i::Index{Lower})
    return @inbounds h_inv*h_inv*apply_stencil(op.closureStencils[Int(i)], v, Int(i))
end

@inline function apply_2nd_derivative(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, i::Index{Interior})
    return @inbounds h_inv*h_inv*apply_stencil(op.innerStencil, v, Int(i))
end

@inline function apply_2nd_derivative(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, i::Index{Upper})
    N = length(v)
    return @inbounds h_inv*h_inv*Int(op.parity)*apply_stencil_backwards(op.closureStencils[N-Int(i)+1], v, Int(i))
end

@inline function apply_2nd_derivative(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, index::Index{Unknown})
    cSize = closuresize(op)
    N = length(v)

    i = Int(index)

    if 0 < i <= cSize
        return apply_2nd_derivative(op, h_inv, v, Index{Lower}(i))
    elseif cSize < i <= N-cSize
        return apply_2nd_derivative(op, h_inv, v, Index{Interior}(i))
    elseif N-cSize < i <= N
        return apply_2nd_derivative(op, h_inv, v, Index{Upper}(i))
    else
        error("Bounds error") # TODO: Make this more standard
    end
end

# Wrapper functions for using regular indecies without specifying regions
@inline function apply_2nd_derivative(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, i::Int)
    return apply_2nd_derivative(op, h_inv, v, Index{Unknown}(i))
end
export apply_2nd_derivative

# TODO: Dispatch on Index{R}?
apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Integer, N::Integer, ::Type{Lower}) where T = v*h*op.quadratureClosure[i]
apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Integer, N::Integer, ::Type{Upper}) where T = v*h*op.quadratureClosure[N-i+1]
apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Integer, N::Integer, ::Type{Interior}) where T = v*h

# TODO: Avoid branching in inner loops
function apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Integer, N::Integer) where T
    r = getregion(i, closuresize(op), N)
    return apply_quadrature(op, h, v, i, N, r)
end
export apply_quadrature

# TODO: Dispatch on Index{R}?
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Integer, N::Integer, ::Type{Lower}) where T = v*h_inv*op.inverseQuadratureClosure[i]
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Integer, N::Integer, ::Type{Upper}) where T = v*h_inv*op.inverseQuadratureClosure[N-i+1]
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Integer, N::Integer, ::Type{Interior}) where T = v*h_inv

# TODO: Avoid branching in inner loops
function apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Integer, N::Integer) where T
    r = getregion(i, closuresize(op), N)
    return apply_inverse_quadrature(op, h_inv, v, i, N, r)
end
export apply_inverse_quadrature

function apply_boundary_value_transpose(op::ConstantStencilOperator, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply_stencil(op.eClosure,v,1)
end

function apply_boundary_value_transpose(op::ConstantStencilOperator, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply_stencil_backwards(op.eClosure,v,length(v))
end
export apply_boundary_value_transpose

function apply_boundary_value(op::ConstantStencilOperator, v::Number, N::Integer, i::Integer, ::Type{Lower})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    op.eClosure[i-1]*v
end

function apply_boundary_value(op::ConstantStencilOperator, v::Number, N::Integer, i::Integer, ::Type{Upper})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    op.eClosure[N-i]*v
end
export apply_boundary_value

function apply_normal_derivative_transpose(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    h_inv*apply_stencil(op.dClosure,v,1)
end

function apply_normal_derivative_transpose(op::ConstantStencilOperator, h_inv::Real, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    -h_inv*apply_stencil_backwards(op.dClosure,v,length(v))
end

export apply_normal_derivative_transpose

function apply_normal_derivative(op::ConstantStencilOperator, h_inv::Real, v::Number, N::Integer, i::Integer, ::Type{Lower})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    h_inv*op.dClosure[i-1]*v
end

function apply_normal_derivative(op::ConstantStencilOperator, h_inv::Real, v::Number, N::Integer, i::Integer, ::Type{Upper})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    -h_inv*op.dClosure[N-i]*v
end

export apply_normal_derivative
