export InverseDiagonalInnerProduct, closuresize
"""
    InverseDiagonalInnerProduct{Dim,T<:Real,M} <: TensorOperator{T,1}

Implements the inverse diagonal inner product operator `Hi` of as a 1D TensorOperator
"""
struct InverseDiagonalInnerProduct{T<:Real,M} <: TensorOperator{T,1}
    h_inv::T # The reciprocl grid spacing could be included in the stencil already. Preferable?
    closure::NTuple{M,T}
    #TODO: Write a nice constructor
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.closure[Int(I)]*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.closure[N-Int(I)+1]v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct,  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closuresize(Hi), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(Hi, v, i)
end

LazyTensors.apply_transpose(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index) where T = LazyTensors.apply(Hi,v,I)


closuresize(Hi::InverseDiagonalInnerProduct{T,M}) where {T,M} =  M
