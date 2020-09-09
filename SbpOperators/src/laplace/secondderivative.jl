"""
    SecondDerivative{T<:Real,N,M,K} <: TensorOperator{T,1}
Implements the Laplace tensor operator `L` with constant grid spacing and coefficients
in 1D dimension
"""

struct SecondDerivative{T,N,M,K} <: TensorOperator{T,1}
    h_inv::T # The grid spacing could be included in the stencil already. Preferable?
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M,Stencil{T,K}}
    parity::Parity
    #TODO: Write a nice constructor
end
export SecondDerivative

LazyTensors.domain_size(D2::SecondDerivative, range_size::NTuple{1,Integer}) = range_size

#TODO: The 1D tensor mappings should not have to dispatch on 1D tuples if we write LazyTensor.apply for vararg right?!?!
#      Currently have to index the Tuple{Index} in each method in order to call the stencil methods which is ugly.
#      I thought I::Vararg{Index,R} fell back to just Index for R = 1

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.closureStencils[Int(I)], v, Int(I))
end

@inline function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.innerStencil, v, Int(I))
end

@inline function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v) # TODO: Use domain_size here instead? N = domain_size(D2,size(v))
    return @inbounds D2.h_inv*D2.h_inv*Int(D2.parity)*apply_stencil_backwards(D2.closureStencils[N-Int(I)+1], v, Int(I))
end

@inline function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v)  # TODO: Use domain_size here instead?
    r = getregion(Int(index), closuresize(D2), N)
    I = Index(Int(index), r)
    return LazyTensors.apply(D2, v, I)
end


@inline function LazyTensors.apply_transpose(D2::SecondDerivative, v::AbstractVector, I::Index)
    return LazyTensors.apply(D2, v, I)
end


function closuresize(D2::SecondDerivative{T,N,M,K}) where {T<:Real,N,M,K}
    return M
end
