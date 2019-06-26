"""
    NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `d` as a TensorMapping
"""
struct NormalDerivative{T,N,M,K} <: TensorMapping{T,2,1}
	op::D2{T,N,M,K}
	grid::EquidistantGrid
	bId::CartesianBoundary
end
export NormalDerivative

# Not correct abstraction level
# TODO: Not type stable D:<
function apply(d::NormalDerivative, v::AbstractArray, I::CartesianIndex{2})
	i = I[dim(d.bId)]
	j = I[3-dim(d.bId)]
	N_i = d.grid.size[dim(d.bId)]

	r = getregion(i, closureSize(d.op), N_i)

	if r != region(d.bId)
		return 0
	end

	if r == Lower
		# Note, closures are indexed by offset. Fix this D:<
		return d.grid.inverse_spacing[dim(d.bId)]*d.op.dClosure[i-1]*v[j]
	elseif r == Upper
		return d.grid.inverse_spacing[dim(d.bId)]*d.op.dClosure[N_i-j]*v[j]
	end
end

function apply_transpose(d::NormalDerivative, v::AbstractArray, I::CartesianIndex{1})
    u = selectdim(v,3-dim(d.bId),I)
    return apply_d(d.op, d.grid.inverse_spacing[dim(d.bId)], u, region(d.bId))
end


"""
    BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}
	op::D2{T,N,M,K}
	grid::EquidistantGrid
	bId::CartesianBoundary
end
export BoundaryValue

# TODO: This is obviouly strange. Is domain_size just discarded? Is there a way to avoid storing grid in BoundaryValue?
# Can we give special treatment to TensorMappings that go to a higher dim?
LazyTensors.range_size(e::BoundaryValue{T}, domain_size::NTuple{1,Integer}) where T = size(e.grid)
LazyTensors.domain_size(e::BoundaryValue{T}, range_size::NTuple{2,Integer}) where T = (range_size[3-dim(e.bId)],);

function LazyTensors.apply(e::BoundaryValue, v::AbstractArray, I::CartesianIndex{2})
	i = I[dim(e.bId)]
	j = I[3-dim(e.bId)]
	N_i = e.grid.size[dim(e.bId)]

	r = getregion(i, closureSize(e.op), N_i)

	if r != region(e.bId)
		return 0
	end

	if r == Lower
		# Note, closures are indexed by offset. Fix this D:<
		return e.op.eClosure[i-1]*v[j]
	elseif r == Upper
		return e.op.eClosure[N_i-j]*v[j]
	end
end

function LazyTensors.apply_transpose(e::BoundaryValue, v::AbstractArray, I::CartesianIndex{1})
	u = selectdim(v,3-dim(e.bId),I)
	return apply_e(e.op, u, region(e.bId))
end



struct Laplace{Dim,T<:Real,N,M,K} <: DiffOpCartesian{Dim}
    grid::EquidistantGrid{Dim,T}
    a::T
    op::D2{Float64,N,M,K}
    # e::BoundaryValue
    # d::NormalDerivative
end

function apply(L::Laplace{Dim}, v::AbstractArray{T,Dim} where T, I::CartesianIndex{Dim}) where Dim
    error("not implemented")
end

# u = L*v
function apply(L::Laplace{1}, v::AbstractVector, i::Int)
    uᵢ = L.a * SbpOperators.apply(L.op, L.grid.spacing[1], v, i)
    return uᵢ
end

@inline function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, I::Tuple{Index{R1}, Index{R2}}) where {R1, R2}
    # 2nd x-derivative
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds uᵢ = L.a*SbpOperators.apply(L.op, L.grid.inverse_spacing[1], vx , I[1])
    # 2nd y-derivative
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds uᵢ += L.a*SbpOperators.apply(L.op, L.grid.inverse_spacing[2], vy, I[2])
    # NOTE: the package qualifier 'SbpOperators' can problably be removed once all "applying" objects use LazyTensors
    return uᵢ
end

# Slow but maybe convenient?
function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, i::CartesianIndex{2})
    I = Index{Unknown}.(Tuple(i))
    apply(L, v, I)
end


struct Neumann{Bid<:BoundaryIdentifier} <: BoundaryCondition end

function sat(L::Laplace{2,T}, bc::Neumann{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, I::CartesianIndex{2}) where {T,Bid}
    e = BoundaryValue(L.op, L.grid, Bid())
    d = NormalDerivative(L.op, L.grid, Bid())
    Hᵧ = BoundaryQuadrature(L.op, L.grid, Bid())
    # TODO: Implement BoundaryQuadrature method

    return -L.Hi*e*Hᵧ*(d'*v - g)
    # Need to handle d'*v - g so that it is an AbstractArray that TensorMappings can act on
end

struct Dirichlet{Bid<:BoundaryIdentifier} <: BoundaryCondition
    tau::Float64
end

function sat(L::Laplace{2,T}, bc::Dirichlet{Bid}, v::AbstractArray{T,2}, g::AbstractVector{T}, i::CartesianIndex{2}) where {T,Bid}
    e = BoundaryValue(L.op, L.grid, Bid())
    d = NormalDerivative(L.op, L.grid, Bid())
    Hᵧ = BoundaryQuadrature(L.op, L.grid, Bid())
    # TODO: Implement BoundaryQuadrature method

    return -L.Hi*(tau/h*e + d)*Hᵧ*(e'*v - g)
    # Need to handle scalar multiplication and addition of TensorMapping
end

# function apply(s::MyWaveEq{D},  v::AbstractArray{T,D}, i::CartesianIndex{D}) where D
# 	return apply(s.L, v, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Lower}}(s.tau),  v, s.g_w, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{1,Upper}}(s.tau),  v, s.g_e, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Lower}}(s.tau),  v, s.g_s, i) +
# 		sat(s.L, Dirichlet{CartesianBoundary{2,Upper}}(s.tau),  v, s.g_n, i)
# end
