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
    h = Grid.spacings(L.grid)[1]
    uᵢ = L.a * apply(L.op, h, v, i)
    return uᵢ
end

using UnsafeArrays

# u = L*v
function apply(L::Laplace{2}, v::AbstractArray{T,2} where T, I::CartesianIndex{2})
    h = Grid.spacings(L.grid)

    # 2nd x-derivative
    @inbounds vx = uview(v, :, I[2])
    @inbounds uᵢ  = apply(L.op, h[1], vx , I[1])
    # 2nd y-derivative
    @inbounds vy = uview(v, I[1], :)
    @inbounds uᵢ += apply(L.op, h[2], vy, I[2])

    return uᵢ
end
