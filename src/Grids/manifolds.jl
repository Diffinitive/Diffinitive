"""
    ParameterSpace{D}

A space of parameters of dimension `D`. Used with `Chart` to indicate which
parameters are valid for that chart.

Common parameter spaces are created using the functions unit sized spaces
* `unitinterval`
* `unitrectangle`
* `unitbox`
* `unittriangle`
* `unittetrahedron`
* `unithyperbox`
* `unitsimplex`

See also: [`Interval`](@ref), [`Rectangle`](@ref), [`Box`](@ref),
[`Triangle`](@ref), [`Tetrahedron`](@ref), [`HyperBox`](@ref),
[`Simplex`](@ref),
"""
abstract type ParameterSpace{D} end

struct HyperBox{T,D} <: ParameterSpace{D}
    a::SVector{D,T}
    b::SVector{D,T}
end

function HyperBox(a,b)
    T = SVector{length(a)}
    HyperBox(convert(T,a), convert(T,b))
end

Interval{T} = HyperBox{T,1}
Rectangle{T} = HyperBox{T,2}
Box{T} = HyperBox{T,3}

limits(box::HyperBox, d) = (box.a[d], box.b[d])
limits(box::HyperBox) = (box.a, box.b)

unitinterval(T=Float64) = unithyperbox(T,1)
unitsquare(T=Float64) = unithyperbox(T,2)
unitcube(T=Float64) = unithyperbox(T,3)
unithyperbox(T, D) = HyperBox((@SVector zeros(T,D)), (@SVector ones(T,D)))
unithyperbox(D) = unithyperbox(Float64,D)


struct Simplex{T,D} <: ParameterSpace{D}
    verticies::NTuple{D,SVector{D,T}}
end

Simplex(verticies::Vararg{AbstractArray}) = Simplex(Tuple(SVector(v...) for v ∈ verticies))

Triangle{T} = Simplex{T,2}
Tetrahedron{T} = Simplex{T,3}

unittriangle(T) = unitsimplex(T,2)
unittetrahedron(T) = unitsimplex(T,3)
function unitsimplex(T,D)
    z = @SVector zeros(T,D)
    unitelement = one(eltype(z))
    verticies = ntuple(i->setindex(z, unitelement, i), 4)
    return Simplex(verticies)
end


"""

A parametrized description of a manifold or part of a manifold.

Should implement a methods for
* `parameterspace`
* `(::Chart)(ξs...)`
"""
abstract type Chart{D} end
# abstract type Chart{D,R} end

domain_dim(::Chart{D}) where D = D
# range_dim(::Chart{D,R}) where {D,R} = R

"""
The parameterspace of a chart
"""
function parameterspace end


# TODO: Add trait for if there is a jacobian available?
# Add package extension to allow calling the getter function anyway if it's not available
# And can we add an informative error that ForwardDiff could be loaded to make it work?
# Or can we handle this be custom implementations? For sometypes in the library it can be implemented explicitly.
# And as an example for ConcreteChart it can be implemented by the user like
# c = ConcreteChart(...)
# jacobian(c::typeof(c)) = ...

struct ConcreteChart{D, PST<:ParameterSpace{D}, MT} <: Chart{D}
    mapping::MT
    parameterspace::PST
end

(c::ConcreteChart)(ξ) = c.mapping(ξ)
parameterspace(c::ConcreteChart) = c.parameterspace

jacobian(c::ConcreteChart, ξ) = jacobian(c.mapping, ξ)

"""
    Atlas

A collection of charts and their connections.
Should implement methods for `charts` and
"""
abstract type Atlas end

"""
    charts(::Atlas)

The colloction of charts in the atlas.
"""
function charts end

"""
    connections

TBD: What exactly should this return?

"""

struct CartesianAtlas <: Atlas
    charts::Matrix{Chart}
end

charts(a::CartesianAtlas) = a.charts

struct UnstructuredAtlas <: Atlas
    charts::Vector{Chart}
    connections
end

charts(a::UnstructuredAtlas) = a.charts


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


struct Circle{T,PT} <: Curve
    c::PT
    r::T
end

(c::Circle)(θ) = c.c + r*@SVector[cos(Θ), sin(Θ)]

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


function polygon_sides(Ps...)
    n = length(Ps)
    return [t->line(t,Ps[i],Ps[mod1(i+1,n)]) for i ∈ eachindex(Ps)]
end
