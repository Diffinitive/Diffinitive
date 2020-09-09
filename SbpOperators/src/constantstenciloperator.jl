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
    N = length(v)
    r = getregion(Int(index), closuresize(op), N)
    i = Index(Int(index), r)
    return apply_2nd_derivative(op, h_inv, v, i)
end
export apply_2nd_derivative

apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Index{Lower}, N::Integer) where T = v*h*op.quadratureClosure[Int(i)]
apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Index{Upper}, N::Integer) where T = v*h*op.quadratureClosure[N-Int(i)+1]
apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, i::Index{Interior}, N::Integer) where T = v*h

function apply_quadrature(op::ConstantStencilOperator, h::Real, v::T, index::Index{Unknown}, N::Integer) where T
    r = getregion(Int(index), closuresize(op), N)
    i = Index(Int(index), r)
    return apply_quadrature(op, h, v, i, N)
end
export apply_quadrature

# TODO: Evaluate if divisions affect performance
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Index{Lower}, N::Integer) where T = h_inv*v/op.quadratureClosure[Int(i)]
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Index{Upper}, N::Integer) where T = h_inv*v/op.quadratureClosure[N-Int(i)+1]
apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, i::Index{Interior}, N::Integer) where T = v*h_inv

function apply_inverse_quadrature(op::ConstantStencilOperator, h_inv::Real, v::T, index::Index{Unknown}, N::Integer) where T
    r = getregion(Int(index), closuresize(op), N)
    i = Index(Int(index), r)
    return apply_inverse_quadrature(op, h_inv, v, i, N)
end

export apply_inverse_quadrature

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

function apply_normal_derivative(op::ConstantStencilOperator, h_inv::Real, v::Number, i::Index, N::Integer, ::Type{Lower})
    @boundscheck if !(0<length(Int(i)) <= N)
        throw(BoundsError())
    end
    h_inv*op.dClosure[Int(i)-1]*v
end

function apply_normal_derivative(op::ConstantStencilOperator, h_inv::Real, v::Number, i::Index, N::Integer, ::Type{Upper})
    @boundscheck if !(0<length(Int(i)) <= N)
        throw(BoundsError())
    end
    -h_inv*op.dClosure[N-Int(i)]*v
end

export apply_normal_derivative
