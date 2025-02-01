"""
    Chart{D}

A parametrized description of a manifold or part of a manifold.
"""
struct Chart{D, PST<:ParameterSpace{D}, MT}
    mapping::MT
    parameterspace::PST
end

Base.ndims(::Chart{D}) where D = D
(c::Chart)(ξ) = c.mapping(ξ)
parameterspace(c::Chart) = c.parameterspace

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
jacobian(c::Chart, ξ) = jacobian(c.mapping, ξ)
# TBD: Can we register a error hint for when jacobian is called with a function that doesn't have a registered jacobian?


# TBD: Should Charts, parameterspaces, Atlases, have boundary names?

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

TBD: What exactly should this return?
"""
function connections end

struct CartesianAtlas <: Atlas
    charts::Matrix{Chart}
end

charts(a::CartesianAtlas) = a.charts
connections(a::CartesianAtlas) = nothing

struct UnstructuredAtlas <: Atlas
    charts::Vector{Chart}
    connections
end

charts(a::UnstructuredAtlas) = a.charts
connections(a::UnstructuredAtlas) = nothing


###
# Geometry
###

abstract type Curve end
abstract type Surface end


struct Line{PT} <: Curve
    p::PT
    tangent::PT
end

(c::Line)(s) = c.p + s*c.tangent


struct LineSegment{PT} <: Curve
    a::PT
    b::PT
end

(c::LineSegment)(s) = (1-s)*c.a + s*c.b


function linesegments(ps...)
    return [LineSegment(ps[i], ps[i+1]) for i ∈ 1:length(ps)-1]
end


function polygon_edges(ps...)
    n = length(ps)
    return [LineSegment(ps[i], ps[mod1(i+1,n)]) for i ∈ eachindex(ps)]
end

struct Circle{T,PT} <: Curve
    c::PT
    r::T
end

function (C::Circle)(θ)
    (;c, r) = C
    c + r*@SVector[cos(θ), sin(θ)]
end

struct TransfiniteInterpolationSurface{T1,T2,T3,T4} <: Surface
    c₁::T1
    c₂::T2
    c₃::T3
    c₄::T4
end

function (s::TransfiniteInterpolationSurface)(u,v)
    c₁, c₂, c₃, c₄ = s.c₁, s.c₂, s.c₃, s.c₄
    P₀₀ = c₁(0)
    P₁₀ = c₂(0)
    P₁₁ = c₃(0)
    P₀₁ = c₄(0)
    return (1-v)*c₁(u) + u*c₂(v) + v*c₃(1-u) + (1-u)*c₄(1-v) - (
        (1-u)*(1-v)*P₀₀ + u*(1-v)*P₁₀ + u*v*P₁₁ + (1-u)*v*P₀₁
    )
end

function (s::TransfiniteInterpolationSurface)(ξ̄::AbstractArray)
    s(ξ̄...)
end

# TODO: Implement jacobian() for the different mapping helpers

