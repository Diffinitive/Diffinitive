"""
     Grid{T,D}

A grid with coordinates of type `T`, e.g. `SVector{3,Float64}`, and dimension
`D`. The grid can be embedded in a higher dimension in which case the number
of indices and the number of components of the coordinate vectors will be
different.

All grids are expected to behave as a grid function for the coordinates.

`Grids` is top level abstract type for grids. A grid should implement Julia's interfaces for
indexing and iteration.

## Note

Importantly a grid does not have to be an `AbstractArray`. The reason is to
allow flexible handling of special types of grids like multi-block grids, or
grids with special indexing.
"""
abstract type Grid{T,D} end

Base.ndims(::Grid{T,D}) where {T,D} = D
Base.eltype(::Type{<:Grid{T}}) where T = T

Base.getindex(g::Grid, I::CartesianIndex) = g[Tuple(I)...]

"""
    coordinate_size(g)

The length of the coordinate vector of `Grid` `g`.
"""
coordinate_size(::Type{<:Grid{T}}) where T = _ncomponents(T)
coordinate_size(g::Grid) = coordinate_size(typeof(g)) # TBD: Name of this function?!

"""
    component_type(gf)

The type of the components of the elements of `gf`. I.e if `gf` is a vector
valued grid function, `component_view(gf)` is the element type of the vectors
at each grid point.

# Examples
```julia-repl
julia> component_type([[1,2], [2,3], [3,4]])
Int64
```
"""
component_type(T::Type) = eltype(eltype(T))
component_type(t) = component_type(typeof(t))

"""
    componentview(gf, component_index...)

A view of `gf` with only the components specified by `component_index...`.

# Examples
```julia-repl
julia> componentview([[1,2], [2,3], [3,4]],2)
3-element ArrayComponentView{Int64, Vector{Int64}, 1, Vector{Vector{Int64}}, Tuple{Int64}}:
 2
 3
 4
```
"""
componentview(gf, component_index...) = ArrayComponentView(gf, component_index)

struct ArrayComponentView{CT,T,D,AT <: AbstractArray{T,D}, IT} <: AbstractArray{CT,D}
    v::AT
    component_index::IT

    function ArrayComponentView(v, component_index)
        CT = typeof(first(v)[component_index...])
        return new{CT, eltype(v), ndims(v), typeof(v), typeof(component_index)}(v,component_index)
    end
end

Base.size(cv) = size(cv.v)
Base.getindex(cv::ArrayComponentView, i::Int) = cv.v[i][cv.component_index...]
Base.getindex(cv::ArrayComponentView, I::Vararg{Int}) = cv.v[I...][cv.component_index...]
IndexStyle(::Type{<:ArrayComponentView{<:Any,<:Any,AT}}) where AT = IndexStyle(AT)

# TODO: Implement `setindex!`?
# TODO: Implement a more general ComponentView that can handle non-AbstractArrays.

"""
    refine(g::Grid, r)

The grid where `g` is refined by the factor `r`.

See also: [`coarsen`](@ref).
"""
function refine end

"""
    coarsen(g::Grid, r)

The grid where `g` is coarsened by the factor `r`.

See also: [`refine`](@ref).
"""
function coarsen end

"""
    boundary_identifiers(g::Grid)

Identifiers for all the boundaries of `g`.
"""
function boundary_identifiers end

"""
    boundary_grid(g::Grid, id::BoundaryIdentifier)

The grid for the boundary specified by `id`.
"""
function boundary_grid end
# TBD: Can we implement a version here that accepts multiple ids and grouped boundaries? Maybe we need multiblock stuff?

"""
    boundary_indices(g::Grid, id::BoundaryIdentifier)

A collection of indices corresponding to the boundary with given id. For grids
with Cartesian indexing these collections will be tuples with elements of type
``Union{Int,Colon}``.

When implementing this method it is expected that the returned collection can
be used to index grid functions to obtain grid functions on the boundary grid.
"""
function boundary_indices end

"""
    eval_on(g::Grid, f)

Lazy evaluation of `f` on the grid. `f` can either be on the form `f(x,y,...)`
with each coordinate as an argument, or on the form `f(x̄)` taking a
coordinate vector.

For concrete array grid functions `map(f,g)` can be used instead.
"""
eval_on(g::Grid, f) = eval_on(g, f, Base.IteratorSize(g))
function eval_on(g::Grid, f, ::Base.HasShape)
    if hasmethod(f, (Any,))
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]), size(g))
    else
        return LazyTensors.LazyFunctionArray((I...)->f(g[I...]...), size(g))
    end
end

"""
    eval_on(g::Grid, f::Number)

Lazy evaluation of a scalar `f` on the grid.
"""
eval_on(g::Grid, f::Number) = return LazyTensors.LazyConstantArray(f, size(g))

_ncomponents(::Type{<:Number}) = 1
_ncomponents(T::Type{<:SVector}) = length(T)


