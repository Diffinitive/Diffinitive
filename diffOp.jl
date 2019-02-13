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

function boundaryCondition(D::DiffOp,b::Grid.BoundaryId,type)::(Closure, Penalty)
    error("not implemented")
end

function interface(Du::DiffOp, Dv::DiffOp, b::Grid.BoundaryId; type)
    error("not implemented")
end

abstract type Closure end

function apply(c::Closure, v::AbstractVector, i::Int)
    error("not implemented")
end

abstract type Penalty end

function apply(c::Penalty, g, i::Int)
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
    for tileaxs ∈ TileIterator(axes(ri), padded_tilesize(T, (5,5), 2)) # TBD: Is this the right way, the right size?
        for j ∈ tileaxs[2], i ∈ tileaxs[1]
            I = ri[i,j]
            u[i,j] = apply(D, v, (Index{r1}(I[1]), Index{r2}(I[2])))
        end
    end
    return nothing
end

function apply(D::DiffOp, v::AbstractVector)::AbstractVector
    u = zeros(eltype(v), size(v))
    apply!(D,v,u)
    return u
end

struct Laplace{Dim,T<:Real,N,M,K} <: DiffOpCartesian{Dim}
    grid::Grid.EquidistantGrid{Dim,T}
    a::T
    op::D2{Float64,N,M,K}
end

function apply(L::Laplace{Dim}, v::AbstractArray{T,Dim} where T, I::CartesianIndex{Dim}) where Dim
    error("not implemented")
end

# u = L*v
function apply(L::Laplace{1}, v::AbstractVector, i::Int)
    uᵢ = L.a * apply(L.op, L.grid.spacing[1], v, i)
    return uᵢ
end

function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, I::Tuple{Index{R1}, Index{R2}}) where {R1, R2}
    # 2nd x-derivative
    @inbounds vx = view(v, :, Int(I[2]))
    @inbounds uᵢ = L.a*apply(L.op, L.grid.spacing[1], vx , I[1])
    # 2nd y-derivative
    @inbounds vy = view(v, Int(I[1]), :)
    @inbounds uᵢ += L.a*apply(L.op, L.grid.spacing[2], vy, I[2])
    return uᵢ
end

# Slow but maybe convenient?
function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, i::CartesianIndex{2})
    I = Index{Unknown}.(Tuple(i))
    apply(L, v, I)
end
