abstract type DiffOp end

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

# DiffOp must have a grid!!!
function apply!(D::DiffOp, u::AbstractVector, v::AbstractVector)
    for i ∈ 1:Grid.numberOfPoints(D.grid)
        u[i] = apply(D, v, i)
    end

    return nothing
end

function apply(D::DiffOp, v::AbstractVector)::AbstractVector
    u = zeros(eltype(v), size(v))
    apply!(D,v,u)
    return u
end

struct Laplace{Dim} <: DiffOp
    grid::Grid.EquidistantGrid{Dim}
    a::Real
    op::D2{Float64}
end

# u = L*v
function apply(L::Laplace{1}, v::AbstractVector, i::Int)
    h = Grid.spacings(L.grid)[1]
    uᵢ = L.a * apply(L.op, h, v, i)
    return uᵢ
end

# u = L*v
function apply(L::Laplace{2}, v::AbstractVector, i::Int)
    h = Grid.spacings(L.grid)

    li = LinearIndices(L.grid.numberOfPointsPerDim)
    ci = CartesianIndices(L.grid.numberOfPointsPerDim)
    I = ci[i]

    # 2nd x-derivative
    uᵢ  = apply(L.op, h[1], view(v, li[:,I[2]]), I[1])
    # 2nd y-derivative
    uᵢ += apply(L.op, h[2], view(v, li[I[1],:]), I[2])

    return uᵢ
end
