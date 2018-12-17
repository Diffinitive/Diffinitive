abstract type DiffOp end

function apply(D::DiffOp, v::AbstractVector)
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
function apply(L::Laplace1D, u::AbstractVector, v::AbstractVector)
    N = closureSize(L.op)
    M = length(v)

    h = scaling(L.grid)

    for i ∈ 1:N
        u[i] = apply(L.op.closureStencils[i], v, i)/h^2
    end

    for i ∈ N+1:M-N
        u[i] = apply(L.op.innerStencil, i)/h^2
    end

    for i ∈ M:-1:M-N+1
        u[i] = apply(flip(L.op.closureStencils[M-i+1]), v, i)/h^2
    end

    return nothing
end
