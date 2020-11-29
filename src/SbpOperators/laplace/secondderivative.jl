"""
    SecondDerivative{T<:Real,N,M,K} <: TensorOperator{T,1}
Implements the Laplace tensor operator `L` with constant grid spacing and coefficients
in 1D dimension
"""

struct SecondDerivative{T,N,M,K} <: TensorMapping{T,1,1}
    h_inv::T # The grid spacing could be included in the stencil already. Preferable?
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M,Stencil{T,K}}
    size::NTuple{1,Int}
end
export SecondDerivative

function SecondDerivative(grid::EquidistantGrid{1}, innerStencil, closureStencils)
    h_inv = inverse_spacing(grid)[1]
    return SecondDerivative(h_inv, innerStencil, closureStencils, size(grid))
end

LazyTensors.range_size(D2::SecondDerivative) = D2.size
LazyTensors.domain_size(D2::SecondDerivative) = D2.size

# Apply for different regions Lower/Interior/Upper or Unknown region
function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.closureStencils[Int(i)], v, Int(i))
end

function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil(D2.innerStencil, v, Int(i))
end

function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v) # TODO: Use domain_size here instead? N = domain_size(D2,size(v))
    return @inbounds D2.h_inv*D2.h_inv*apply_stencil_backwards(D2.closureStencils[N-Int(i)+1], v, Int(i))
end

function LazyTensors.apply(D2::SecondDerivative{T}, v::AbstractVector{T}, i) where T
    N = length(v)  # TODO: Use domain_size here instead?
    r = getregion(i, closuresize(D2), N)
    return LazyTensors.apply(D2, v, Index(i, r))
end

closuresize(D2::SecondDerivative{T,N,M,K}) where {T<:Real,N,M,K} = M
