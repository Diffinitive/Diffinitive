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

The lenght of the coordinate vector of `Grid` `g`.
"""
coordinate_size(::Type{<:Grid{T}}) where T = _ncomponents(T)
coordinate_size(g::Grid) = coordinate_size(typeof(g)) # TBD: Name of this function?!

"""
    component_type(gf)

The type of the components of the elements of `gf`.
"""
component_type(T::Type) = eltype(eltype(T))
component_type(t) = component_type(typeof(t))

componentview(gf, component_index...) = ArrayComponentView(gf, component_index)

# REVIEW: Should this only be defined for vector-valued component types of the same dimension?
# Now one can for instance do:  v = [rand(2,2),rand(2,2), rand(2,1)] and cv = componentview(v,1,2)
# resulting in #undef in the third component of cv.
# RESPONSE: I don't think it's possible to stop the user from
# doing stupid things. My inclination is to just keep it simple and let the
# user read the error messages they get.
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
Base.getindex(cv::ArrayComponentView, I::Vararg{Int}) = cv.v[I...][cv.component_index...] #REVIEW: Will this allocate if I... slices v? if so, we should probably use a view on v?
# RESPONSE: I imagine the values of `cv` will be small static vectors most of
# the time for which this won't be a problem. I say we cross that bridge when
# there is an obvoius need. (Just slapping a @view on there seems to be
# changing the return tyoe to a 0-dimensional array. That's where i gave up.)
IndexStyle(::Type{<:ArrayComponentView{<:Any,<:Any,AT}}) where AT = IndexStyle(AT)

# TODO: Implement the remaining optional methods from the array interface
# setindex!(A, v, i::Int)
# setindex!(A, v, I::Vararg{Int, N})
# iterate
# length(A)
# similar(A)
# similar(A, ::Type{S})
# similar(A, dims::Dims)
# similar(A, ::Type{S}, dims::Dims)
# # Non-traditional indices
# axes(A)
# similar(A, ::Type{S}, inds)
# similar(T::Union{Type,Function}, inds)

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


