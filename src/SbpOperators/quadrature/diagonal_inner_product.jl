export DiagonalInnerProduct, closuresize
"""
    DiagonalInnerProduct{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the diagnoal norm operator `H` of Dim dimension as a TensorMapping
"""
struct DiagonalInnerProduct{T,M} <: TensorMapping{T,1,1}
    h::T
    quadratureClosure::NTuple{M,T}
    size::Tuple{Int}
end

function DiagonalInnerProduct(g::EquidistantGrid{1}, quadratureClosure)
    return DiagonalInnerProduct(spacing(g)[1], quadratureClosure, size(g))
end

LazyTensors.range_size(H::DiagonalInnerProduct) = H.size
LazyTensors.domain_size(H::DiagonalInnerProduct) = H.size

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds H.h*H.quadratureClosure[Int(i)]*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.quadratureClosure[N-Int(i)+1]*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds H.h*v[Int(i)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},  v::AbstractVector{T}, i) where T
    N = length(v);
    r = getregion(i, closuresize(H), N)
    return LazyTensors.apply(H, v, Index(i, r))
end

LazyTensors.apply_transpose(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, i) where T = LazyTensors.apply(H,v,i)

closuresize(H::DiagonalInnerProduct{T,M}) where {T,M} = M
