"""
inverse_diagonal_quadrature(g,quadrature_closure)

Constructs the diagonal quadrature inverse operator `Hi` on a grid of `Dim` dimensions as
a `TensorMapping`. The one-dimensional operator is a InverseDiagonalQuadrature, while
the multi-dimensional operator is the outer-product of the one-dimensional operators
in each coordinate direction.
"""
function inverse_diagonal_quadrature(g::EquidistantGrid{Dim}, quadrature_closure) where Dim
    Hi = InverseDiagonalQuadrature(restrict(g,1), quadrature_closure)
    for i ∈ 2:Dim
        Hi = Hi⊗InverseDiagonalQuadrature(restrict(g,i), quadrature_closure)
    end
    return Hi
end
export inverse_diagonal_quadrature


"""
    InverseDiagonalQuadrature{T,M} <: TensorMapping{T,1,1}

Implements the one-dimensional inverse diagonal quadrature operator as a `TensorMapping
TODO: Elaborate on properties
"""
struct InverseDiagonalQuadrature{T<:Real,M} <: TensorMapping{T,1,1}
    h_inv::T
    closure::NTuple{M,T}
    size::Tuple{Int}
end
export InverseDiagonalQuadrature

function InverseDiagonalQuadrature(g::EquidistantGrid{1}, quadrature_closure)
    return InverseDiagonalQuadrature(inverse_spacing(g)[1], 1 ./ quadrature_closure, size(g))
end


LazyTensors.range_size(Hi::InverseDiagonalQuadrature) = Hi.size
LazyTensors.domain_size(Hi::InverseDiagonalQuadrature) = Hi.size


function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, I::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.closure[Int(I)]*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, I::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.closure[N-Int(I)+1]*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, I::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(I)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature,  v::AbstractVector{T}, index::Index{Unknown}) where T
    N = length(v);
    r = getregion(Int(index), closure_size(Hi), N)
    i = Index(Int(index), r)
    return LazyTensors.apply(Hi, v, i)
end

LazyTensors.apply_transpose(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, I::Index) where T = LazyTensors.apply(Hi,v,I)

"""
    closure_size(H)
Returns the size of the closure stencil of a InverseDiagonalQuadrature `Hi`.
"""
closure_size(Hi::InverseDiagonalQuadrature{T,M}) where {T,M} =  M
export closure_size
