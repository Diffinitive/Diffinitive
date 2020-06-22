#TODO: Naming?! What is this? It is a 1D tensor operator but what is then the
# potentially multi-D laplace tensor mapping then?
# Ideally I would like the below to be the laplace operator in 1D, while the
# multi-D operator is a a tuple of the 1D-operator. Possible via recursive
# definitions? Or just bad design?
"""
    ConstantLaplaceOperator{T<:Real,N,M,K} <: TensorOperator{T,1}
Implements the Laplace tensor operator `L` with constant grid spacing and coefficients
in 1D dimension
"""
struct ConstantLaplaceOperator{T<:Real,N,M,K} <: TensorOperator{T,1}
    h_inv::T # The grid spacing could be included in the stencil already. Preferable?
    a::T # TODO: Better name?
    innerStencil::Stencil{T,N}
    closureStencils::NTuple{M,Stencil{T,K}}
    parity::Parity
    #TODO: Write a nice constructor
end

@enum Parity begin
    odd = -1
    even = 1
end

LazyTensors.domain_size(L::ConstantLaplaceOperator, range_size::NTuple{1,Integer}) = range_size

function LazyTensors.apply(L::ConstantLaplaceOperator{T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    return apply(L, v, I[1])
end

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function LazyTensors.apply(L::ConstantLaplaceOperator, v::AbstractVector, i::Index{Lower})
    return @inbounds L.a*L.h_inv*L.h_inv*apply_stencil(L.closureStencils[Int(i)], v, Int(i))
end

@inline function LazyTensors.apply(L::ConstantLaplaceOperator, v::AbstractVector, i::Index{Interior})
    return @inbounds L.a*L.h_inv*L.h_inv*apply_stencil(L.innerStencil, v, Int(i))
end

@inline function LazyTensors.apply(L::ConstantLaplaceOperator, v::AbstractVector, i::Index{Upper})
    N = length(v) # TODO: Use domain_size here instead?
    return @inbounds L.a*L.h_inv*L.h_inv*Int(L.parity)*apply_stencil_backwards(L.closureStencils[N-Int(i)+1], v, Int(i))
end

@inline function LazyTensors.apply(L::ConstantLaplaceOperator, v::AbstractVector, index::Index{Unknown})
    N = length(v)  # TODO: Use domain_size here instead?
    r = getregion(Int(index), closuresize(L), N)
    i = Index(Int(index), r)
    return apply(L, v, i)
end

function closuresize(L::ConstantLaplaceOperator{T<:Real,N,M,K}) where T,N,M,K
    return M
end
