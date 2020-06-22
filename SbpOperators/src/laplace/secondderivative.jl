"""
    SecondDerivative{T<:Real,N,M,K} <: TensorOperator{T,1}
Implements the Laplace tensor operator `L` with constant grid spacing and coefficients
in 1D dimension
"""
struct SecondDerivative{T<:Real,N,M,K} <: TensorOperator{T,1}
    h_inv::T # The grid spacing could be included in the stencil already. Preferable?
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M,Stencil{T,K}}
    parity::Parity
    #TODO: Write a nice constructor
end

@enum Parity begin
    odd = -1
    even = 1
end

LazyTensors.domain_size(D2::SecondDerivative, range_size::NTuple{1,Integer}) = range_size

function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    return apply(D2, v, I[1])
end

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function LazyTensors.apply(D2::SecondDerivative, v::AbstractVector, i::Index{Lower})
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.closureStencils[Int(i)], v, Int(i))
end

@inline function LazyTensors.apply(D2::SecondDerivative, v::AbstractVector, i::Index{Interior})
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.innerStencil, v, Int(i))
end

@inline function LazyTensors.apply(D2::SecondDerivative, v::AbstractVector, i::Index{Upper})
    N = length(v) # TODO: Use domain_size here instead?
    return @inbounds D2.h_inv*D2.h_inv*Int(D2.parity)*apply_stencil_backwards(D2.closureStencils[N-Int(i)+1], v, Int(i))
end

@inline function LazyTensors.apply(D2::SecondDerivative, v::AbstractVector, index::Index{Unknown})
    N = length(v)  # TODO: Use domain_size here instead?
    r = getregion(Int(index), closuresize(L), N)
    i = Index(Int(index), r)
    return apply(D2, v, i)
end

function closuresize(D2::SecondDerivative{T<:Real,N,M,K}) where T,N,M,K
    return M
end
