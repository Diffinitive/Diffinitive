export D2, closuresize, readOperator, apply_e, apply_d, apply_e_T, apply_d_T

@enum Parity begin
    odd = -1
    even = 1
end

struct D2{T,N,M,K} <: ConstantStencilOperator
    quadratureClosure::NTuple{M,T}
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M,Stencil{T,K}}
    eClosure::Stencil{T,M}
    dClosure::Stencil{T,M}
    parity::Parity
end

function closuresize(D::D2)::Int
    return length(D.quadratureClosure)
end

apply_quadrature(op::D2{T}, h::Real, v::T, i::Integer, N::Integer, ::Type{Lower}) where T = v*h*op.quadratureClosure[i]
apply_quadrature(op::D2{T}, h::Real, v::T, i::Integer, N::Integer, ::Type{Upper}) where T = v*h*op.quadratureClosure[N-i+1]
apply_quadrature(op::D2{T}, h::Real, v::T, i::Integer, N::Integer, ::Type{Interior}) where T = v*h

function apply_quadrature(op::D2{T}, h::Real, v::T, i::Integer, N::Integer) where T
    r = getregion(i, closuresize(op), N)
    return apply_quadrature(op, h, v, i, N, r)
end
export apply_quadrature

function apply_e_T(op::D2, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply(op.eClosure,v,1)
end

function apply_e_T(op::D2, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply(flip(op.eClosure),v,length(v))
end


function apply_e(op::D2, v::Number, N::Integer, i::Integer, ::Type{Lower})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    op.eClosure[i-1]*v
end

function apply_e(op::D2, v::Number, N::Integer, i::Integer, ::Type{Upper})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    op.eClosure[N-i]*v
end

function apply_d_T(op::D2, h_inv::Real, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    h_inv*apply(op.dClosure,v,1)
end

function apply_d_T(op::D2, h_inv::Real, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    -h_inv*apply(flip(op.dClosure),v,length(v))
end

function apply_d(op::D2, h_inv::Real, v::Number, N::Integer, i::Integer, ::Type{Lower})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    h_inv*op.dClosure[i-1]*v
end

function apply_d(op::D2, h_inv::Real, v::Number, N::Integer, i::Integer, ::Type{Upper})
    @boundscheck if !(0<length(i) <= N)
        throw(BoundsError())
    end
    -h_inv*op.dClosure[N-i]*v
end
