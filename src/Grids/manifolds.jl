"""
    Chart{D}

A parametrized description of a manifold or part of a manifold.
"""
struct Chart{D, PST<:ParameterSpace{D}, MT}
    mapping::MT
    parameterspace::PST
end

Base.ndims(::Chart{D}) where D = D
parameterspace(c::Chart) = c.parameterspace

function (c::Chart)(ξ)
    if ξ ∉ parameterspace(c)
        throw(DomainError(ξ, "chart was called logical coordinates outside the parameterspace. If this was inteded, use the `mapping` field from the Chart struct instead."))
    end
    return c.mapping(ξ)
end

"""
    jacobian(c::Chart, ξ)

The jacobian of the mapping evaluated at `ξ`. This defers to the
implementation of `jacobian` for the mapping itself. If no implementation is
available one can easily be specified for either the mapping function or the
chart itself.
```julia
c = Chart(f, ps)
jacobian(f::typeof(f), ξ) = f′(ξ)
```
or
```julia
c = Chart(f, ps)
jacobian(c::typeof(c),ξ) = f′(ξ)
```
which will both allow calling `jacobian(c,ξ)`.
"""
function jacobian(c::Chart, ξ)
    if ξ ∉ parameterspace(c)
        throw(DomainError(ξ, "jacobian was called with logical coordinates outside the parameterspace of the chart. If this was inteded, use the `mapping` field from the Chart struct instead."))
    end
    return jacobian(c.mapping, ξ)
end

boundary_identifiers(c::Chart) = boundary_identifiers(parameterspace(c))

function normal(c::Chart, boundary, ξ)
    # The formula is based on expressing the normal in terms of vectors ∂x/∂ξᵢ,
    # call the coordinate vector a.
    # In physical coordinates we have n = ∂x/∂ξᵢaᵢ.
    # For a boundary where ξₖ = const, n should be orthogonal to ∂x/∂ξⱼ for all j != k
    # This gives the system
    #    ∂x/∂ξⱼ ⋅ ∂x/∂ξᵢaᵢ = δⱼₖ
    #    ⇔ gᵢⱼaᵢ = δⱼₖ
    #    ⇔ aᵢ = gⁱʲδⱼₖ
    #    ⇔ n = ∂x/∂ξᵢ gⁱʲδⱼₖ

    ∂x∂ξ = jacobian(c, ξ)
    g = ∂x∂ξ' * ∂x∂ξ
    g⁻¹ = inv(g)
    σ = _boundary_sign(eltype(g), boundary)

    k = grid_id(boundary)
    n = ∂x∂ξ * g⁻¹[:, k]
    return σ * n / norm(n)
end


"""
    Atlas

A collection of charts and their connections.
Should implement methods for `charts` and `connections`.
"""
abstract type Atlas end

"""
    charts(::Atlas)

The colloction of charts in the atlas.
"""
function charts end

"""
    connections(::Atlas)

Collection of 2-tuples of multiblock boundary identifiers.
"""
function connections end


"""
    CartesianAtlas{D,C<:Chart,AT<:AbstractArray{C,D}} <: Atlas

An atlas where the charts are arranged and connected like an array.
"""
struct CartesianAtlas{D,C<:Chart,AT<:AbstractArray{C,D}} <: Atlas
    charts::AT
end

charts(a::CartesianAtlas) = a.charts

function connections(a::CartesianAtlas)
    c = Tuple{MultiBlockBoundary, MultiBlockBoundary}[]

    for d ∈ 1:ndims(charts(a))
        Is = eachslice(CartesianIndices(charts(a)); dims=d)
        for i ∈ 1:length(Is)-1 # For each interface between slices
            for jk ∈ eachindex(Is[i]) # For each block in slice
                Iᵢⱼₖ = Tuple(Is[i][jk])
                Iᵢ₊₁ⱼₖ = Tuple(Is[i+1][jk])
                push!(c,
                    (
                        MultiBlockBoundary{Iᵢⱼₖ,   CartesianBoundary{d,UpperBoundary}}(),
                        MultiBlockBoundary{Iᵢ₊₁ⱼₖ, CartesianBoundary{d,LowerBoundary}}(),
                    )
                )
            end
        end
    end

    return c
end

"""
    boundary_identifiers(a::CartesianAtlas)

All non-connected boundaries of the charts of `a`.
"""
function boundary_identifiers(a::CartesianAtlas)
    bs = MultiBlockBoundary[]

    for d ∈ 1:ndims(charts(a))
        Is = eachslice(CartesianIndices(charts(a)); dims=d)

        for (i,b) ∈ ((1,LowerBoundary),(length(Is),UpperBoundary)) # For first and last slice
            for jk ∈ eachindex(Is[i]) # For each block in slice
                Iᵢⱼₖ = Tuple(Is[i][jk])
                push!(bs,
                    MultiBlockBoundary{Iᵢⱼₖ,   CartesianBoundary{d,b}}(),
                )
            end
        end
    end

    return bs
end


"""
    UnstructuredAtlas{C<:Chart, CN<:Tuple{MultiBlockBoundary,MultiBlockBoundary}, ...} <: Atlas

An atlas with connections determined by a vector `MultiBlockBoundary` pairs.
"""
struct UnstructuredAtlas{C<:Chart, CN<:Tuple{MultiBlockBoundary,MultiBlockBoundary}, CV<:AbstractVector{C}, CNV<:AbstractVector{CN}} <: Atlas
    charts::CV
    connections::CNV
end

charts(a::UnstructuredAtlas) = a.charts
connections(a::UnstructuredAtlas) = a.connections

"""
    boundary_identifiers(a::UnstructuredAtlas)

All non-connected boundaries of the charts of `a`.
"""
function boundary_identifiers(a::UnstructuredAtlas)
    bs = MultiBlockBoundary[]

    for (i,c) ∈ enumerate(charts(a))
        for b ∈ boundary_identifiers(c)
            mbb = MultiBlockBoundary{i,typeof(b)}()

            if !any(cn->mbb∈cn, connections(a))
                push!(bs, mbb)
            end
        end
    end

    return bs
end

"""
    FunctionWithJacobian{FT,JT}
    FunctionWithJacobian(f,J)

Wraps a function and its jacobian to make it available for Grids.jacobian

See also: [with_jacobian](@ref)
"""
struct FunctionWithJacobian{FT,JT}
    f::FT
    J::JT
end

(fJ::FunctionWithJacobian)(x) = fJ.f(x)
jacobian(fJ::FunctionWithJacobian, x) = fJ.J(x)


"""
    with_jacobian(f, Jfun)

Create a FunctionWithJacobian from `f` using `J(x) = Jfun(f,x)`.

# Example
```julia-repl
julia> using ForwardDiff, StaticArrays
julia> f = with_jacobian(ForwardDiff.jacobian) do ξ
    @SVector[ξ[1], ξ[2]*(ξ[1]^2+1)]
end;

julia> f([1,2])
2-element SVector{2, Int64} with indices SOneTo(2):
 1
 4

julia> jacobian(f, [1,2])
2×2 Matrix{Int64}:
 1  0
 4  2

```
"""
with_jacobian(f, Jfun) = FunctionWithJacobian(f, x->Jfun(f,x))

"""
    with_jacobian(x, pm::ParameterSpace, Jfun)

Create a Chart from `x(ξ)` and `pm` using `J(ξ) = Jfun(f,ξ)`.

# Example
```julia-repl
julia> using ForwardDiff, StaticArrays

julia> c = with_jacobian(unitsquare(), ForwardDiff.jacobian) do ξ
           @SVector[ξ[1], ξ[2]*(ξ[1]^2+1)]
       end;

julia> c([1,1/2])
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 1.0

julia> jacobian(c,[1,1/2])
2×2 Matrix{Float64}:
 1.0  0.0
 1.0  2.0
```
"""
with_jacobian(x, pm::ParameterSpace, Jfun) = Chart(with_jacobian(x,Jfun), pm)

"""
    with_jacobian(xJ, pm::ParameterSpace)

Create a Chart with `pm` and mapping + jacobian from `xJ`. `xJ(ξ)` should return
a tuple `x(ξ), J(ξ)`

# Example
```julia-repl
julia> using ForwardDiff, StaticArrays

julia> c = with_jacobian(unitsquare()) do ξ
           x = @SVector[ξ[1], ξ[2]*(ξ[1]^2+1)]
           J = @SMatrix[
               1          0       ;
               2ξ[1]*ξ[2] ξ[1]^2+1;
           ]

           (x,J)
       end;

julia> c([1,1/2])
2-element SVector{2, Float64} with indices SOneTo(2):
 1.0
 1.0

julia> jacobian(c,[1,1/2])
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 1.0  0.0
 1.0  2.0
```
"""
function with_jacobian(xJ, pm::ParameterSpace)
    _check_coordinates_and_jacobian(xJ, centroid(pm))

    x(ξ) = xJ(ξ)[1]
    J(ξ) = xJ(ξ)[2]

    return Chart(FunctionWithJacobian(x,J), pm)
end

function _check_coordinates_and_jacobian(xJ, ξ)
    x_J = xJ(ξ)

    if !(x_J isa Tuple && length(x_J) == 2)
        throw(ArgumentError("with_jacobian(xJ, pm) expects xJ(ξ) to return a 2-tuple `(x, J)`."))
    end

    return nothing
end
