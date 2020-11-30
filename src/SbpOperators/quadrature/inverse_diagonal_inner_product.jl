export InverseDiagonalInnerProduct, closuresize
"""
    InverseDiagonalInnerProduct{Dim,T<:Real,M} <: TensorMapping{T,1,1}

Implements the inverse diagonal inner product operator `Hi` of as a 1D TensorOperator
"""
struct InverseDiagonalInnerProduct{T<:Real,M} <: TensorMapping{T,1,1}
    h_inv::T
    inverseQuadratureClosure::NTuple{M,T}
    size::Tuple{Int}
end

function InverseDiagonalInnerProduct(g::EquidistantGrid{1}, quadratureClosure)
    return InverseDiagonalInnerProduct(inverse_spacing(g)[1], 1 ./ quadratureClosure, size(g))
end

LazyTensors.range_size(Hi::InverseDiagonalInnerProduct) = Hi.size
LazyTensors.domain_size(Hi::InverseDiagonalInnerProduct) = Hi.size


function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.inverseQuadratureClosure[Int(i)]*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.inverseQuadratureClosure[N-Int(i)+1]*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalInnerProduct{T},  v::AbstractVector{T}, i) where T
    N = length(v);
    r = getregion(i, closuresize(Hi), N)
    return LazyTensors.apply(Hi, v, Index(i, r))
end

LazyTensors.apply_transpose(Hi::InverseDiagonalInnerProduct{T}, v::AbstractVector{T}, i) where T = LazyTensors.apply(Hi,v,i)


closuresize(Hi::InverseDiagonalInnerProduct{T,M}) where {T,M} =  M
