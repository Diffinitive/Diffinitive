struct Line{PT}
    p::PT
    tangent::PT

    Line{PT}(p::PT, tangent::PT) where PT = new{PT}(p,tangent)
end

function Line(p, t)
    p = SVector{length(p)}(p)
    t = SVector{length(t)}(t)
    p, t = promote(p, t)

    return Line{typeof(p)}(p,t)
end

(c::Line)(s) = c.p + s*c.tangent

Grids.jacobian(l::Line, t) = l.tangent

struct LineSegment{PT}
    a::PT
    b::PT

    LineSegment{PT}(p::PT, tangent::PT) where PT = new{PT}(p,tangent)
end

function LineSegment(a, b)
    a = SVector{length(a)}(a)
    b = SVector{length(b)}(b)
    a, b = promote(a, b)

    return LineSegment{typeof(a)}(a,b)
end

(c::LineSegment)(s) = (1-s)*c.a + s*c.b

Grids.jacobian(c::LineSegment, s) = c.b - c.a

function linesegments(ps...)
    return [LineSegment(ps[i], ps[i+1]) for i ∈ 1:length(ps)-1]
end


function polygon_edges(ps...)
    n = length(ps)
    return [LineSegment(ps[i], ps[mod1(i+1,n)]) for i ∈ eachindex(ps)]
end

struct Circle{PT,T}
    c::PT
    r::T

    Circle{PT,T}(c,r) where {PT,T} = new{PT,T}(c,r)
end

function Circle(c,r)
    c = SVector{2}(c)
    return Circle{typeof(c), typeof(r)}(c,r)
end

function (C::Circle)(θ)
    (;c, r) = C
    c + r*@SVector[cos(θ), sin(θ)]
end

function Grids.jacobian(C::Circle, θ)
    (;r) = C
    r*@SVector[-sin(θ), cos(θ)]
end

struct TransfiniteInterpolationSurface{T1,T2,T3,T4}
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

function check_transfiniteinterpolation(s::TransfiniteInterpolationSurface)
    if check_transfiniteinterpolation(Bool, s)
        return nothing
    else
        error("The end of each curve in the transfinite interpolation should be the same as the beginning of the next curve.")
    end
end

function check_transfiniteinterpolation(::Type{Bool}, s::TransfiniteInterpolationSurface)
    if !isapprox(s.c₁(1), s.c₂(0))
        return false
    end

    if !isapprox(s.c₂(1), s.c₃(0))
        return false
    end

    if !isapprox(s.c₃(1), s.c₄(0))
        return false
    end

    if !isapprox(s.c₄(1), s.c₁(0))
        return false
    end

    return true
end

# TODO: Implement jacobian() for the different mapping helpers
# TODO: Add doc strings
