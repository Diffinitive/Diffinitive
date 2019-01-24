abstract type DiffOp end

function apply!(D::DiffOp, u::AbstractVector, v::AbstractVector)
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

# Differential operator for a*d^2/dx^2
struct Laplace{D, T<:Real} <: DiffOp
    grid::Grid.EquidistantGrid{D,T}
    a::T
    op::D2{Float64}
end

# u = L*v
function apply!(L::Laplace{1}, u::AbstractVector, v::AbstractVector)
    h = Grid.spacings(L.grid)[1]
    apply!(L.op, u, v, h)
    u .= L.a * u
    return nothing
end

# u = L*v
function apply!(L::Laplace{2}, u::AbstractVector, v::AbstractVector)
    u .= 0*u
    h = Grid.spacings(L.grid)

    li = LinearIndices(L.grid.numberOfPointsPerDim)
    n_x, n_y = L.grid.numberOfPointsPerDim


    # For each x
    temp = zeros(eltype(u), n_y)
    for i ∈ 1:n_x

        v_i = view(v, li[i,:])
        apply!(L.op, temp, v_i, h[2])

        u[li[i,:]] += temp
    end

    # For each y
    temp = zeros(eltype(u), n_x)
    for i ∈ 1:n_y
        v_i = view(v, li[:,i])
        apply!(L.op, temp, v_i, h[1])

        u[li[:,i]] += temp
    end

    u .= L.a*u

    return nothing
end
