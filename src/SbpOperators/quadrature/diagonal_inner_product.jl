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

# Behövs den här?
function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index) where T
    return @inbounds apply(H, v, I)
end

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds H.h*H.quadratureClosure[Int(I)]*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds H.h*H.quadratureClosure[N-Int(I)+1]*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds H.h*v[Int(I)]
end

function LazyTensors.apply(H::DiagonalInnerProduct{T},  v::AbstractVector{T}, i) where T
    N = length(v);
    r = getregion(i, closuresize(H), N)
    I = Index(i, r)
    return LazyTensors.apply(H, v, I)
end

LazyTensors.apply_transpose(H::DiagonalInnerProduct{T}, v::AbstractVector{T}, i) where T = LazyTensors.apply(H,v,i)

closuresize(H::DiagonalInnerProduct{T,M}) where {T,M} = M
