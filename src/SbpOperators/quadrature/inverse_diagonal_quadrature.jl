"""
inverse_diagonal_quadrature(g,quadrature_closure)

Constructs the inverse `Hi` of a `DiagonalQuadrature` on a grid of `Dim` dimensions as
a `TensorMapping`. The one-dimensional operator is a `InverseDiagonalQuadrature`, while
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

Implements the inverse of a one-dimensional `DiagonalQuadrature` as a `TensorMapping`
The operator is defined by the reciprocal of the quadrature interval length `h_inv`, the
reciprocal of the quadrature closure weights `closure` and the number of quadrature intervals `size`. The
interior stencil has the weight 1.
"""
struct InverseDiagonalQuadrature{T<:Real,M} <: TensorMapping{T,1,1}
    h_inv::T
    closure::NTuple{M,T}
    size::Tuple{Int}
end
export InverseDiagonalQuadrature

"""
    InverseDiagonalQuadrature(g, quadrature_closure)

Constructs the `InverseDiagonalQuadrature` on the `EquidistantGrid` `g` with
closure given by the reciprocal of `quadrature_closure`.
"""
function InverseDiagonalQuadrature(g::EquidistantGrid{1}, quadrature_closure)
    return InverseDiagonalQuadrature(inverse_spacing(g)[1], 1 ./ quadrature_closure, size(g))
end

"""
    domain_size(Hi::InverseDiagonalQuadrature)

The size of an object in the range of `Hi`
"""
LazyTensors.range_size(Hi::InverseDiagonalQuadrature) = Hi.size

"""
    domain_size(Hi::InverseDiagonalQuadrature)

The size of an object in the domain of `Hi`
"""
LazyTensors.domain_size(Hi::InverseDiagonalQuadrature) = Hi.size

"""
    apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, i) where T
Implements the application `(Hi*v)[i]` an `Index{R}` where `R` is one of the regions
`Lower`,`Interior`,`Upper`. If `i` is another type of index (e.g an `Int`) it will first
be converted to an `Index{R}`.
"""
function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, i::Index{Lower}) where T
    return @inbounds Hi.h_inv*Hi.closure[Int(i)]*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, i::Index{Upper}) where T
    N = length(v);
    return @inbounds Hi.h_inv*Hi.closure[N-Int(i)+1]*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, i::Index{Interior}) where T
    return @inbounds Hi.h_inv*v[Int(i)]
end

function LazyTensors.apply(Hi::InverseDiagonalQuadrature{T},  v::AbstractVector{T}, i) where T
    N = length(v);
    r = getregion(i, closure_size(Hi), N)
    return LazyTensors.apply(Hi, v, Index(i, r))
end

LazyTensors.apply_transpose(Hi::InverseDiagonalQuadrature{T}, v::AbstractVector{T}, i) where T = LazyTensors.apply(Hi,v,i)

"""
    closure_size(Hi)
Returns the size of the closure stencil of a InverseDiagonalQuadrature `Hi`.
"""
closure_size(Hi::InverseDiagonalQuadrature{T,M}) where {T,M} =  M
