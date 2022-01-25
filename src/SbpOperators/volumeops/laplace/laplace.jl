# REVIEW: The style of name `Laplace` might clash with other concepts. When
# thinking about implementing the variable second derivative I think I will
# have to create it as a full TM for the full dimensional problem instead of
# building it as a 1D operator and then use that with outer products. The
# natural name there would be `VariableSecondDerivative` (or something
# similar). But the similarity of the two names would suggest that `Laplace`
# and `VariableSecondDerivative` are the same kind of thing, which they
# wouldn't be.
#
# How do we distinguish the kind of type we are implementing here and what we
# could potentially do for the variable second derivative?
#
# I see two ways out:
#   * Come up with a name for these sets of operators and change `Laplace` accordingly.
#   * Come up with a name for the bare operators and change `VariableSecondDerivative` accordingly.

"""
    Laplace{T, Dim, TMdiffop} <: TensorMapping{T,Dim,Dim}
    Laplace(grid, filename; order)

Implements the Laplace operator, approximating ∑d²/xᵢ² , i = 1,...,`Dim` as a
`TensorMapping`. Additionally, `Laplace` stores the inner product and boundary
operators relevant for constructing a SBP finite difference scheme as a `TensorMapping`.

`Laplace(grid, filename; order)` creates the Laplace operator defined on `grid`,
where the operators are read from TOML. The differential operator is created
using `laplace(grid,...)`.

Note that all properties of Laplace, excluding the differential operator `Laplace.D`, are
abstract types. For performance reasons, they should therefore be
accessed via the provided getter functions (e.g `inner_product(::Laplace)`).

"""
struct Laplace{T, Dim, TMdiffop<:TensorMapping{T,Dim,Dim}} <: TensorMapping{T,Dim,Dim}
    D::TMdiffop # Differential operator
    H::TensorMapping # Inner product operator
    H_inv::TensorMapping # Inverse inner product operator
    e::StaticDict{<:BoundaryIdentifier,<:TensorMapping} # Boundary restriction operators.
    d::StaticDict{<:BoundaryIdentifier,<:TensorMapping} # Normal derivative operators
    H_boundary::StaticDict{<:BoundaryIdentifier,<:TensorMapping} # Boundary quadrature operators
end
export Laplace

function Laplace(grid, filename; order)
    
    # Read stencils
    stencil_set = read_stencil_set(filename; order)
    # TODO: Removed once we can construct the volume and
    # boundary operators by op(grid, read_stencil_set(fn; order,...)).
    D_inner_stecil = parse_stencil(stencil_set["D2"]["inner_stencil"])
    D_closure_stencils = parse_stencil.(stencil_set["D2"]["closure_stencils"])
    H_inner_stencils = parse_scalar(stencil_set["H"]["inner"])
    H_closure_stencils = parse_tuple(stencil_set["H"]["closure"])
    e_closure_stencil = parse_stencil(stencil_set["e"]["closure"])
    d_closure_stencil = parse_stencil(stencil_set["d1"]["closure"])
    # REVIEW: Do we add the methods to get rid of this in this branch or a new one?

    # Volume operators
    Δ =  laplace(grid, D_inner_stecil, D_closure_stencils)
    H =  inner_product(grid, H_inner_stencils, H_closure_stencils)
    H⁻¹ = inverse_inner_product(grid, H_inner_stencils, H_closure_stencils)

    # Boundary operator - id pairs
    ids = boundary_identifiers(grid)
    # REVIEW: Change suggestion: Seems more readable to me but pretty subjective so feel free to ignore
    e_pairs  = map(id -> Pair(id, boundary_restriction(grid, e_closure_stencil, id)),                            ids)
    d_pairs  = map(id -> Pair(id, normal_derivative(grid, d_closure_stencil, id)),                               ids)
    Hᵧ_pairs = map(id -> Pair(id, inner_product(boundary_grid(grid, id), H_inner_stencils, H_closure_stencils)), ids)

    return Laplace(Δ, H, H⁻¹, StaticDict(e_pairs), StaticDict(d_pairs), StaticDict(Hᵧ_pairs))
end

# TODO: Consider pretty printing of the following form
# Base.show(io::IO, L::Laplace{T, Dim}) where {T,Dim,TM} =  print(io, "Laplace{$T, $Dim, $TM}(", L.D, L.H, L.H_inv, L.e, L.d, L.H_boundary, ")")
# REVIEW: Should leave a todo here to update this once we have some pretty printing for tensor mappings in general.

LazyTensors.range_size(L::Laplace) = LazyTensors.range_size(L.D)
LazyTensors.domain_size(L::Laplace) = LazyTensors.domain_size(L.D)
LazyTensors.apply(L::Laplace, v::AbstractArray, I...) = LazyTensors.apply(L.D,v,I...)


"""
    inner_product(L::Laplace)

Returns the inner product operator associated with `L`

"""
inner_product(L::Laplace) = L.H
export inner_product


"""
    inverse_inner_product(L::Laplace)

Returns the inverse of the inner product operator associated with `L`

"""
inverse_inner_product(L::Laplace) = L.H_inv
export inverse_inner_product


"""
    boundary_restriction(L::Laplace, id::BoundaryIdentifier)
    boundary_restriction(L::Laplace, ids::Tuple)
    boundary_restriction(L::Laplace, ids...)

Returns boundary restriction operator(s) associated with `L` for the boundary(s)
identified by id(s).

"""
boundary_restriction(L::Laplace, id::BoundaryIdentifier) = L.e[id]
boundary_restriction(L::Laplace, ids::Tuple) = map(id-> L.e[id], ids)
boundary_restriction(L::Laplace, ids...) = boundary_restriction(L, ids)
# REVIEW: I propose changing the following implementations according to the
# above. There are some things we're missing with regards to
# `BoundaryIdentifier`, for example we should be able to handle groups of
# boundaries as a single `BoundaryIdentifier`. I don't know if we can figure
# out the interface for that now or if we save it for the future but either
# way these methods will be affected.

export boundary_restriction


"""
    normal_derivative(L::Laplace, id::BoundaryIdentifier)
    normal_derivative(L::Laplace, ids::NTuple{N,BoundaryIdentifier})
    normal_derivative(L::Laplace, ids...)

Returns normal derivative operator(s) associated with `L` for the boundary(s)
identified by id(s).

"""
normal_derivative(L::Laplace, id::BoundaryIdentifier) = L.d[id]
normal_derivative(L::Laplace, ids::NTuple{N,BoundaryIdentifier}) where N = ntuple(i->L.d[ids[i]],N)
normal_derivative(L::Laplace, ids::Vararg{BoundaryIdentifier,N}) where N = ntuple(i->L.d[ids[i]],N)
export normal_derivative


"""
    boundary_quadrature(L::Laplace, id::BoundaryIdentifier)
    boundary_quadrature(L::Laplace, ids::NTuple{N,BoundaryIdentifier})
    boundary_quadrature(L::Laplace, ids...)

Returns boundary quadrature operator(s) associated with `L` for the boundary(s)
identified by id(s).

"""
boundary_quadrature(L::Laplace, id::BoundaryIdentifier) = L.H_boundary[id]
boundary_quadrature(L::Laplace, ids::NTuple{N,BoundaryIdentifier}) where N = ntuple(i->L.H_boundary[ids[i]],N)
boundary_quadrature(L::Laplace, ids::Vararg{BoundaryIdentifier,N}) where N = ntuple(i->L.H_boundary[ids[i]],N)
export boundary_quadrature


"""
    laplace(grid::EquidistantGrid{Dim}, inner_stencil, closure_stencils)

Creates the Laplace operator operator `Δ` as a `TensorMapping`

`Δ` approximates the Laplace operator ∑d²/xᵢ² , i = 1,...,`Dim` on `grid`, using
the stencil `inner_stencil` in the interior and a set of stencils `closure_stencils`
for the points in the closure regions.

On a one-dimensional `grid`, `Δ` is equivalent to `second_derivative`. On a
multi-dimensional `grid`, `Δ` is the sum of multi-dimensional `second_derivative`s
where the sum is carried out lazily.
"""
function laplace(grid::EquidistantGrid, inner_stencil, closure_stencils)
    Δ = second_derivative(grid, inner_stencil, closure_stencils, 1)
    for d = 2:dimension(grid)
        Δ += second_derivative(grid, inner_stencil, closure_stencils, d)
    end
    return Δ
end
export laplace
