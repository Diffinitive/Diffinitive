export DiagonalInnerProduct, closuresize
"""
    DiagonalInnerProduct{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the diagnoal norm operator `H` of Dim dimension as a TensorMapping
"""
struct DiagonalInnerProduct{T,M} <: TensorOperator{T,1}
    h::T # The grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

LazyTensors.domain_size(H::DiagonalInnerProduct, range_size::NTuple{1,Integer}) = range_size

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index) where T
    return @inbounds apply(H, v, I)
end

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds H.h*H.closure[Int(I)]*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.closure[N-Int(I)+1]v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds H.h*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(H), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(H, v, i)
end

LazyTensors.apply_transpose(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index) where T = LazyTensors.apply(H,v,I)

closuresize(H::DiagonalInnerProduct{T,M}) where {T,M} = M
