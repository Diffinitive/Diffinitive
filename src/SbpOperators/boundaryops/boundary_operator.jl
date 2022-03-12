"""
    boundary_operator(grid,closure_stencil,boundary)

Creates a boundary operator on a `Dim`-dimensional grid for the
specified `boundary`. The action of the operator is determined by `closure_stencil`.

When `Dim=1`, the corresponding `BoundaryOperator` tensor mapping is returned.
When `Dim>1`, the `BoundaryOperator` `op` is inflated by the outer product
of `IdentityMappings` in orthogonal coordinate directions, e.g for `Dim=3`,
the boundary restriction operator in the y-direction direction is `Ix⊗op⊗Iz`.
"""
function boundary_operator(grid::EquidistantGrid, closure_stencil, boundary::CartesianBoundary)
    #TODO:Check that dim(boundary) <= Dim?

    # Create 1D boundary operator
    r = region(boundary)
    d = dim(boundary)
    op = BoundaryOperator(restrict(grid, d), closure_stencil, r)

    # Create 1D IdentityMappings for each coordinate direction
    one_d_grids = restrict.(Ref(grid), Tuple(1:dimension(grid)))
    Is = IdentityMapping{eltype(grid)}.(size.(one_d_grids))

    # Formulate the correct outer product sequence of the identity mappings and
    # the boundary operator
    parts = Base.setindex(Is, op, d)
    return foldl(⊗, parts)
end

"""
    BoundaryOperator{T,R,N} <: TensorMapping{T,0,1}

Implements the boundary operator `op` for 1D as a `TensorMapping`

`op` is the restriction of a grid function to the boundary using some closure `Stencil{T,N}`.
The boundary to restrict to is determined by `R`.
`op'` is the prolongation of a zero dimensional array to the whole grid using the same closure stencil.
"""
struct BoundaryOperator{T,R<:Region,N} <: TensorMapping{T,0,1}
    stencil::Stencil{T,N}
    size::Int
end

BoundaryOperator{R}(stencil::Stencil{T,N}, size::Int) where {T,R,N} = BoundaryOperator{T,R,N}(stencil, size)

"""
    BoundaryOperator(grid::EquidistantGrid{1}, closure_stencil, region)

Constructs the BoundaryOperator with stencil `closure_stencil` for a one-dimensional `grid`, restricting to
to the boundary specified by `region`.
"""
function BoundaryOperator(grid::EquidistantGrid{1}, closure_stencil::Stencil{T,N}, region::Region) where {T,N}
    return BoundaryOperator{T,typeof(region),N}(closure_stencil,size(grid)[1])
end

"""
    closure_size(::BoundaryOperator)
The size of the closure stencil.
"""
closure_size(::BoundaryOperator{T,R,N}) where {T,R,N} = N

LazyTensors.range_size(op::BoundaryOperator) = ()
LazyTensors.domain_size(op::BoundaryOperator) = (op.size,)

function LazyTensors.apply(op::BoundaryOperator{<:Any,Lower}, v::AbstractVector)
    apply_stencil(op.stencil,v,1)
end

function LazyTensors.apply(op::BoundaryOperator{<:Any,Upper}, v::AbstractVector)
    apply_stencil_backwards(op.stencil,v,op.size)
end

function LazyTensors.apply_transpose(op::BoundaryOperator{<:Any,Lower}, v::AbstractArray{<:Any,0}, i::Index{Lower})
    return op.stencil[Int(i)-1]*v[]
end

function LazyTensors.apply_transpose(op::BoundaryOperator{<:Any,Upper}, v::AbstractArray{<:Any,0}, i::Index{Upper})
    return op.stencil[op.size[1] - Int(i)]*v[]
end

# Catch all combinations of Lower, Upper and Interior not caught by the two previous methods.
function LazyTensors.apply_transpose(op::BoundaryOperator, v::AbstractArray{<:Any,0}, i::Index)
    return zero(eltype(v))
end

function LazyTensors.apply_transpose(op::BoundaryOperator, v::AbstractArray{<:Any,0}, i)
    r = getregion(i, closure_size(op), op.size)
    apply_transpose(op, v, Index(i,r))
end
