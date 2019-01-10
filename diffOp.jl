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

function interface(Du::DiffOp, Dv::DiffOp, b::grid.BoundaryId; type)
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
    N = closureSize(L.op)
    M = length(v)

    h = scaling(L.grid)

    apply!(L.op, u, v, grid.spacings(L.grid)[1], 1, L.grid.numberOfPointsPerDim, stride=1)
    return nothing
end
