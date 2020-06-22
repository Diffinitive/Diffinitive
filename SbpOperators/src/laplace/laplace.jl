"""
    Laplace{Dim,T<:Real,N,M,K} <: TensorOperator{T,Dim}

Implements the Laplace operator `L` in Dim dimensions as a tensor operator
The multi-dimensional tensor operator simply consists of a tuple of the 1D
Laplace tensor operator as defined by ConstantLaplaceOperator.
"""
struct Laplace{Dim,T<:Real,N,M,K} <: TensorOperator{T,Dim}
    D2::NTuple(Dim,SecondDerivative{T,N,M,K})
    #TODO: Write a good constructor
end
export Laplace

LazyTensors.domain_size(H::Laplace{Dim}, range_size::NTuple{Dim,Integer}) = range_size

function LazyTensors.apply(L::Laplace{Dim,T}, v::AbstractArray{T,Dim}, I::NTuple{Dim,Index}) where {T,Dim}
    error("not implemented")
end

# u = L*v
function LazyTensors.apply(L::Laplace{1,T}, v::AbstractVector{T}, I::NTuple{1,Index}) where T
    return apply(L.D2[1],v,I)
end


@inline function LazyTensors.apply(L::Laplace{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T
    # 2nd x-derivative
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds uᵢ = apply(L.D2[1], vx , (I[1],)) #Tuple conversion here is ugly. How to do it? Should we use indexing here?

    # 2nd y-derivative
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds uᵢ += apply(L.D2[2], vy , (I[2],)) #Tuple conversion here is ugly. How to do it?

    return uᵢ
end

quadrature(L::Laplace) = Quadrature(L.op, L.grid)
inverse_quadrature(L::Laplace) = InverseQuadrature(L.op, L.grid)
boundary_value(L::Laplace, bId::CartesianBoundary) = BoundaryValue(L.op, L.grid, bId)
normal_derivative(L::Laplace, bId::CartesianBoundary) = NormalDerivative(L.op, L.grid, bId)
boundary_quadrature(L::Laplace, bId::CartesianBoundary) = BoundaryQuadrature(L.op, L.grid, bId)
export quadrature

# At the moment the grid property is used all over. It could possibly be removed if we implement all the 1D operators as TensorMappings
"""
    Quadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the quadrature operator `H` of Dim dimension as a TensorMapping
"""
struct Quadrature{Dim,T<:Real,N,M,K} <: TensorOperator{T,Dim}
    op::D2{T,N,M,K}
    grid::EquidistantGrid{Dim,T}
end
export Quadrature

LazyTensors.domain_size(H::Quadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

@inline function LazyTensors.apply(H::Quadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T
    N = size(H.grid)
    # Quadrature in x direction
    @inbounds q = apply_quadrature(H.op, spacing(H.grid)[1], v[I] , I[1], N[1])
    # Quadrature in y-direction
    @inbounds q = apply_quadrature(H.op, spacing(H.grid)[2], q, I[2], N[2])
    return q
end

LazyTensors.apply_transpose(H::Quadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T = LazyTensors.apply(H,v,I)


"""
    InverseQuadrature{Dim,T<:Real,N,M,K} <: TensorMapping{T,Dim,Dim}

Implements the inverse quadrature operator `inv(H)` of Dim dimension as a TensorMapping
"""
struct InverseQuadrature{Dim,T<:Real,N,M,K} <: TensorOperator{T,Dim}
    op::D2{T,N,M,K}
    grid::EquidistantGrid{Dim,T}
end
export InverseQuadrature

LazyTensors.domain_size(H_inv::InverseQuadrature{Dim}, range_size::NTuple{Dim,Integer}) where Dim = range_size

@inline function LazyTensors.apply(H_inv::InverseQuadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T
    N = size(H_inv.grid)
    # Inverse quadrature in x direction
    @inbounds q_inv = apply_inverse_quadrature(H_inv.op, inverse_spacing(H_inv.grid)[1], v[I] , I[1], N[1])
    # Inverse quadrature in y-direction
    @inbounds q_inv = apply_inverse_quadrature(H_inv.op, inverse_spacing(H_inv.grid)[2], q_inv, I[2], N[2])
    return q_inv
end

LazyTensors.apply_transpose(H_inv::InverseQuadrature{2,T}, v::AbstractArray{T,2}, I::NTuple{2,Index}) where T = LazyTensors.apply(H_inv,v,I)

"""
    BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}
    op::D2{T,N,M,K}
    grid::EquidistantGrid{2}
    bId::CartesianBoundary
end
export BoundaryValue

# TODO: This is obviouly strange. Is domain_size just discarded? Is there a way to avoid storing grid in BoundaryValue?
# Can we give special treatment to TensorMappings that go to a higher dim?
function LazyTensors.range_size(e::BoundaryValue{T}, domain_size::NTuple{1,Integer}) where T
    if dim(e.bId) == 1
        return (UnknownDim, domain_size[1])
    elseif dim(e.bId) == 2
        return (domain_size[1], UnknownDim)
    end
end
LazyTensors.domain_size(e::BoundaryValue{T}, range_size::NTuple{2,Integer}) where T = (range_size[3-dim(e.bId)],)
# TODO: Make a nicer solution for 3-dim(e.bId)

# TODO: Make this independent of dimension
function LazyTensors.apply(e::BoundaryValue{T}, v::AbstractArray{T}, I::NTuple{2,Index}) where T
    i = I[dim(e.bId)]
    j = I[3-dim(e.bId)]
    N_i = size(e.grid)[dim(e.bId)]
    return apply_boundary_value(e.op, v[j], i, N_i, region(e.bId))
end

function LazyTensors.apply_transpose(e::BoundaryValue{T}, v::AbstractArray{T}, I::NTuple{1,Index}) where T
    u = selectdim(v,3-dim(e.bId),Int(I[1]))
    return apply_boundary_value_transpose(e.op, u, region(e.bId))
end

"""
    NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `d` as a TensorMapping
"""
struct NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}
    op::D2{T,N,M,K}
    grid::EquidistantGrid{2}
    bId::CartesianBoundary
end
export NormalDerivative

# TODO: This is obviouly strange. Is domain_size just discarded? Is there a way to avoid storing grid in BoundaryValue?
# Can we give special treatment to TensorMappings that go to a higher dim?
function LazyTensors.range_size(e::NormalDerivative, domain_size::NTuple{1,Integer})
    if dim(e.bId) == 1
        return (UnknownDim, domain_size[1])
    elseif dim(e.bId) == 2
        return (domain_size[1], UnknownDim)
    end
end
LazyTensors.domain_size(e::NormalDerivative, range_size::NTuple{2,Integer}) = (range_size[3-dim(e.bId)],)

# TODO: Not type stable D:<
# TODO: Make this independent of dimension
function LazyTensors.apply(d::NormalDerivative{T}, v::AbstractArray{T}, I::NTuple{2,Index}) where T
    i = I[dim(d.bId)]
    j = I[3-dim(d.bId)]
    N_i = size(d.grid)[dim(d.bId)]
    h_inv = inverse_spacing(d.grid)[dim(d.bId)]
    return apply_normal_derivative(d.op, h_inv, v[j], i, N_i, region(d.bId))
end

function LazyTensors.apply_transpose(d::NormalDerivative{T}, v::AbstractArray{T}, I::NTuple{1,Index}) where T
    u = selectdim(v,3-dim(d.bId),Int(I[1]))
    return apply_normal_derivative_transpose(d.op, inverse_spacing(d.grid)[dim(d.bId)], u, region(d.bId))
end

"""
    BoundaryQuadrature{T,N,M,K} <: TensorOperator{T,1}

Implements the boundary operator `q` as a TensorOperator
"""
struct BoundaryQuadrature{T,N,M,K} <: TensorOperator{T,1}
    op::D2{T,N,M,K}
    grid::EquidistantGrid{2}
    bId::CartesianBoundary
end
export BoundaryQuadrature

# TODO: Make this independent of dimension
function LazyTensors.apply(q::BoundaryQuadrature{T}, v::AbstractArray{T,1}, I::NTuple{1,Index}) where T
    h = spacing(q.grid)[3-dim(q.bId)]
    N = size(v)
    return apply_quadrature(q.op, h, v[I[1]], I[1], N[1])
end

LazyTensors.apply_transpose(q::BoundaryQuadrature{T}, v::AbstractArray{T,1}, I::NTuple{1,Index}) where T = LazyTensors.apply(q,v,I)




struct Neumann{Bid<:BoundaryIdentifier} <: BoundaryCondition end

function sat(L::Laplace{2,T}, bc::Neumann{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, I::CartesianIndex{2}) where {T,Bid}
    e = boundary_value(L, Bid())
    d = normal_derivative(L, Bid())
    Hᵧ = boundary_quadrature(L, Bid())
    H⁻¹ = inverse_quadrature(L)
    return (-H⁻¹*e*Hᵧ*(d'*v - g))[I]
end

struct Dirichlet{Bid<:BoundaryIdentifier} <: BoundaryCondition
    tau::Float64
end

function sat(L::Laplace{2,T}, bc::Dirichlet{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, i::CartesianIndex{2}) where {T,Bid}
    e = boundary_value(L, Bid())
    d = normal_derivative(L, Bid())
    Hᵧ = boundary_quadrature(L, Bid())
    H⁻¹ = inverse_quadrature(L)
    return (-H⁻¹*(tau/h*e + d)*Hᵧ*(e'*v - g))[I]
    # Need to handle scalar multiplication and addition of TensorMapping
end

# function apply(s::MyWaveEq{D},  v::AbstractArray{T,D}, i::CartesianIndex{D}) where D
    #   return apply(s.L, v, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Lower}}(s.tau),  v, s.g_w, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Upper}}(s.tau),  v, s.g_e, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Lower}}(s.tau),  v, s.g_s, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Upper}}(s.tau),  v, s.g_n, i)
# end
