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

function boundaryCondition(D::DiffOp)
    error("not implemented")
end

function interface(Du::DiffOp, Dv::DiffOp, b::Grid.BoundaryId; type)
    error("not implemented")
end


# Differential operator for a*d^2/dx^2
struct Laplace1D <: DiffOp
    grid
    a
    op
end

# u = L*v
function apply!(L::Laplace1D, u::AbstractVector, v::AbstractVector)
    h = Grid.spacings(L.grid)[1]
    apply!(L.op, u, v, h)
    u .= L.a * u
    return nothing
end


# Differential operator for a*d^2/dx^2 + a*d^2/dy^2
struct Laplace2D <: DiffOp
    grid
    a
    op
end

# u = L*v
function apply!(L::Laplace2D, u::AbstractVector, v::AbstractVector)
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
