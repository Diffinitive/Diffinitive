abstract type DiffOp end

# TBD: The "error("not implemented")" thing seems to be hiding good error information. How to fix that? Different way of saying that these should be implemented?
function apply(D::DiffOp, v::AbstractVector, i::Int)
    error("not implemented")
end

function innerProduct(D::DiffOp, u::AbstractVector, v::AbstractVector)::Real
    error("not implemented")
end

function matrixRepresentation(D::DiffOp)
    error("not implemented")
end

abstract type DiffOpCartesian{Dim} <: DiffOp end

# DiffOp must have a grid of dimension Dim!!!
function apply!(D::DiffOpCartesian{Dim}, u::AbstractArray{T,Dim}, v::AbstractArray{T,Dim}) where {T,Dim}
    for I ∈ eachindex(D.grid)
        u[I] = apply(D, v, I)
    end

    return nothing
end

function apply_region!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}) where T
    apply_region!(D, u, v, Lower, Lower)
    apply_region!(D, u, v, Lower, Interior)
    apply_region!(D, u, v, Lower, Upper)
    apply_region!(D, u, v, Interior, Lower)
    apply_region!(D, u, v, Interior, Interior)
    apply_region!(D, u, v, Interior, Upper)
    apply_region!(D, u, v, Upper, Lower)
    apply_region!(D, u, v, Upper, Interior)
    apply_region!(D, u, v, Upper, Upper)
    return nothing
end

# Maybe this should be split according to b3fbef345810 after all?! Seems like it makes performance more predictable
function apply_region!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}, r1::Type{<:Region}, r2::Type{<:Region}) where T
    for I ∈ regionindices(D.grid.size, closureSize(D.op), (r1,r2))
        @inbounds indextuple = (Index{r1}(I[1]), Index{r2}(I[2]))
        @inbounds u[I] = apply(D, v, indextuple)
    end
    return nothing
end

function apply_tiled!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}) where T
    apply_region_tiled!(D, u, v, Lower, Lower)
    apply_region_tiled!(D, u, v, Lower, Interior)
    apply_region_tiled!(D, u, v, Lower, Upper)
    apply_region_tiled!(D, u, v, Interior, Lower)
    apply_region_tiled!(D, u, v, Interior, Interior)
    apply_region_tiled!(D, u, v, Interior, Upper)
    apply_region_tiled!(D, u, v, Upper, Lower)
    apply_region_tiled!(D, u, v, Upper, Interior)
    apply_region_tiled!(D, u, v, Upper, Upper)
    return nothing
end

using TiledIteration
function apply_region_tiled!(D::DiffOpCartesian{2}, u::AbstractArray{T,2}, v::AbstractArray{T,2}, r1::Type{<:Region}, r2::Type{<:Region}) where T
    ri = regionindices(D.grid.size, closureSize(D.op), (r1,r2))
    # TODO: Pass Tilesize to function
    for tileaxs ∈ TileIterator(axes(ri), padded_tilesize(T, (5,5), 2))
        for j ∈ tileaxs[2], i ∈ tileaxs[1]
            I = ri[i,j]
            u[I] = apply(D, v, (Index{r1}(I[1]), Index{r2}(I[2])))
        end
    end
    return nothing
end

function apply(D::DiffOp, v::AbstractVector)::AbstractVector
    u = zeros(eltype(v), size(v))
    apply!(D,v,u)
    return u
end

struct NormalDerivative{N,M,K}
	op::D2{Float64,N,M,K}
	grid::EquidistantGrid
	bId::CartesianBoundary
end

function apply_transpose(d::NormalDerivative, v::AbstractArray, I::Integer)
	u = selectdim(v,3-dim(d.bId),I)
	return apply_d(d.op, d.grid.inverse_spacing[dim(d.bId)], u, region(d.bId))
end

# Not correct abstraction level
# TODO: Not type stable D:<
function apply(d::NormalDerivative, v::AbstractArray, I::Tuple{Integer,Integer})
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

struct BoundaryValue{N,M,K}
	op::D2{Float64,N,M,K}
	grid::EquidistantGrid
	bId::CartesianBoundary
end

function apply(e::BoundaryValue, v::AbstractArray, I::Tuple{Integer,Integer})
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

function apply_transpose(e::BoundaryValue, v::AbstractArray, I::Integer)
	u = selectdim(v,3-dim(e.bId),I)
	return apply_e(e.op, u, region(e.bId))
end

struct Laplace{Dim,T<:Real,N,M,K} <: DiffOpCartesian{Dim}
    grid::EquidistantGrid{Dim,T}
    a::T
    op::D2{Float64,N,M,K}
    e::BoundaryValue
    d::NormalDerivative
end

function apply(L::Laplace{Dim}, v::AbstractArray{T,Dim} where T, I::CartesianIndex{Dim}) where Dim
    error("not implemented")
end

# u = L*v
function apply(L::Laplace{1}, v::AbstractVector, i::Int)
    uᵢ = L.a * apply(L.op, L.grid.spacing[1], v, i)
    return uᵢ
end

@inline function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, I::Tuple{Index{R1}, Index{R2}}) where {R1, R2}
    # 2nd x-derivative
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds uᵢ = L.a*apply(L.op, L.grid.inverse_spacing[1], vx , I[1])
    # 2nd y-derivative
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds uᵢ += L.a*apply(L.op, L.grid.inverse_spacing[2], vy, I[2])
    return uᵢ
end

# Slow but maybe convenient?
function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, i::CartesianIndex{2})
    I = Index{Unknown}.(Tuple(i))
    apply(L, v, I)
end

struct BoundaryOperator

end


"""
A BoundaryCondition should implement the method
    sat(::DiffOp, v::AbstractArray, data::AbstractArray, ...)
"""
abstract type BoundaryCondition end

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
