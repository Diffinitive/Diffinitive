export D2, closureSize, readOperator, apply_e, apply_d, apply_e_T, apply_d_T

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

function closureSize(D::D2)::Int
    return length(D.quadratureClosure)
end

function apply_e_T(op::D2, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closureSize(op)
        throw(BoundsError())
    end
    apply(op.eClosure,v,1)
end

function apply_e_T(op::D2, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closureSize(op)
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
    @boundscheck if length(v) < closureSize(op)
        throw(BoundsError())
    end
    h_inv*apply(op.dClosure,v,1)
end

function apply_d_T(op::D2, h_inv::Real, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closureSize(op)
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
