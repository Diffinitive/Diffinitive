struct Line{PT}
    p::PT
    tangent::PT

    Line{PT}(p::PT, tangent::PT) where PT = new{PT}(p,tangent)
end


"""
    Line(p,t)

A line, as a callable object, starting at `p` with tangent `t`.
The parametrization is ``l(s) = p + st``.

# Example
```julia-repl
julia> l = Grids.Line([1,1],[2,1])
Diffinitive.Grids.Line{StaticArraysCore.SVector{2, Int64}}([1, 1], [2, 1])

julia> l(0)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 1

julia> l(1)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 3
 2
```

See also: [`LineSegment`](@ref).
"""
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


"""
    LineSegment(a,b)

A line segment, as a callable object, from `a` to `b`.
The parametrization is ``l(s) = (1-s)a + s*b``.

# Example
```julia-repl
julia> l = Grids.LineSegment([1,1],[2,1])
Diffinitive.Grids.LineSegment{StaticArraysCore.SVector{2, Int64}}([1, 1], [2, 1])

julia> l(0)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 1
 1

julia> l(0.5)
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 1.5
 1.0

julia> l(1)
2-element StaticArraysCore.SVector{2, Int64} with indices SOneTo(2):
 2
 1
```

See also: [`Line`](@ref).
"""
function LineSegment(a, b)
    a = SVector{length(a)}(a)
    b = SVector{length(b)}(b)
    a, b = promote(a, b)

    return LineSegment{typeof(a)}(a,b)
end

(c::LineSegment)(s) = (1-s)*c.a + s*c.b

Grids.jacobian(c::LineSegment, s) = c.b - c.a


"""
    linesegments(ps...)

An array of line segments between the points `ps[1]`, `ps[2]`, and so on.

See also: [`polygon_edges`](@ref).
"""
function linesegments(ps...)
    return [LineSegment(ps[i], ps[i+1]) for i ∈ 1:length(ps)-1]
end


"""
    polygon_edges(ps...)

An array of line segments between the points `ps[1]`, `ps[2]`, and so on
including the segment between `ps[end]` and `ps[1]`.

See also: [`linesegments`](@ref).
"""
function polygon_edges(ps...)
    n = length(ps)
    return [LineSegment(ps[i], ps[mod1(i+1,n)]) for i ∈ eachindex(ps)]
end


struct Circle{PT,T}
    c::PT
    r::T

    Circle{PT,T}(c,r) where {PT,T} = new{PT,T}(c,r)
end

"""
    Circle(c,r)

A circle with center `c` and radius `r` paramatrized with the angle to the x-axis.

# Example
```julia-repl
julia> c = Grids.Circle([1,1], 2)
Diffinitive.Grids.Circle{StaticArraysCore.SVector{2, Int64}, Int64}([1, 1], 2)

julia> c(0)
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 3.0
 1.0

julia> c(π/2)
2-element StaticArraysCore.SVector{2, Float64} with indices SOneTo(2):
 1.0000000000000002
 3.0
```
"""
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


struct Arc{PT,T}
    c::Circle{PT,T}
    θ₀::T
    θ₁::T
end

"""
    Arc(C::Circle, θ₀, θ₁)

A circular arc as a callable object. The arc is around the circle `C` between
angles `θ₀` and `θ₁` and is paramatrized between 0 and 1.

See also: [`arc`](@ref), [`Circle`](@ref).
"""
function Arc(C, θ₀, θ₁)
    r, θ₀, θ₁ = promote(C.r, θ₀, θ₁)

    return Arc(Circle(C.c, r), θ₀, θ₁)
end

function (A::Arc)(t)
    (; θ₀, θ₁) = A
    return A.c((1-t)*θ₀ + t*θ₁)
end

function Grids.jacobian(A::Arc, t)
    (;c, θ₀, θ₁) = A
    return (θ₁-θ₀)*jacobian(c, t)
end


"""
    arc(a,b,r)

A circular arc between the points `a` and `b` with radius `abs(r)`. If `r > 0`
the arc goes counter clockwise and if `r<0` the arc goes clockwise. The arc is
parametrized such that if `A = arc(a,b,r)` then `A(0)` corresponds to `a` and
`A(1)` to `b`.

See also: [`Arc`](@ref), [`Circle`](@ref).
"""
function arc(a,b,r)
    if abs(r) < norm(b-a)/2
        throw(DomainError(r, "arc was called with radius r = $r smaller than half the distance between the points."))
    end

    R̂ = @SMatrix[0 -1; 1 0]

    α = sign(r)*√(r^2 - norm((b-a)/2)^2)
    t̂ = R̂*(b-a)/norm(b-a)

    c = (a+b)/2 + α*t̂

    ca = a-c
    cb = b-c
    θₐ = atan(ca[2],ca[1])
    θᵦ = atan(cb[2],cb[1])

    Δθ = mod(θᵦ-θₐ+π, 2π)-π # Δθ in the interval (-π,π)

    if r > 0
        Δθ = abs(Δθ)
    else
        Δθ = -abs(Δθ)
    end

    return Arc(Circle(c,abs(r)), θₐ, θₐ+Δθ)
end

"""
    TransfiniteInterpolationSurface(c₁, c₂, c₃, c₄)

A surface defined by the transfinite interpolation of the curves `c₁`, `c₂`, `c₃`, and  `c₄`.
The transfinite interpolation maps the unit square ([0,1]⊗[0,1]) to the patch delimited by the given curves.
The curves should encircle the patch counterclockwise.

See https://en.wikipedia.org/wiki/Transfinite_interpolation for more information on transfinite interpolation.
"""
struct TransfiniteInterpolationSurface{T1,T2,T3,T4}
    c₁::T1
    c₂::T2
    c₃::T3
    c₄::T4
end

function (s::TransfiniteInterpolationSurface)(u,v)
    if (u,v) ∉ unitsquare()
        throw(DomainError((u,v), "Transfinite interpolation was called with parameters outside the unit square."))
    end
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

"""
    check_transfiniteinterpolation(s::TransfiniteInterpolationSurface)

Throw an error if the ends of the curves in the transfinite interpolation do not match.
"""
function check_transfiniteinterpolation(s::TransfiniteInterpolationSurface)
    if check_transfiniteinterpolation(Bool, s)
        return nothing
    else
        throw(ArgumentError("The end of each curve in the transfinite interpolation should be the same as the beginning of the next curve."))
    end
end

"""
    check_transfiniteinterpolation(Bool, s::TransfiniteInterpolationSurface)

Return true if the ends of the curves in the transfinite interpolation match.
"""
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

function Grids.jacobian(s::TransfiniteInterpolationSurface, ξ̄)
    if ξ̄ ∉ unitsquare()
        throw(DomainError(ξ̄, "Transfinite interpolation was called with parameters outside the unit square."))
    end
    u, v = ξ̄

    c₁, c₂, c₃, c₄ = s.c₁, s.c₂, s.c₃, s.c₄
    P₀₀ = c₁(0)
    P₁₀ = c₂(0)
    P₁₁ = c₃(0)
    P₀₁ = c₄(0)

    ∂x̄∂ξ₁ = (1-v)*jacobian(c₁,u) + c₂(v) - v*jacobian(c₃,1-u) -c₄(1-v) - (
        -(1-v)*P₀₀ + (1-v)*P₁₀ + v*P₁₁ - v*P₀₁
    )

    ∂x̄∂ξ₂ = -c₁(u) + u*jacobian(c₂,v) + c₃(1-u) - (1-u)*jacobian(c₄,1-v) - (
        -(1-u)*P₀₀ - u*P₁₀ + u*P₁₁ + (1-u)*P₀₁
    )

    return [∂x̄∂ξ₁ ∂x̄∂ξ₂]
end
