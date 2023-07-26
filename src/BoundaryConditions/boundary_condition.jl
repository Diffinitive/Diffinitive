# TODO:  Should BoundaryData just be used for traits
# of the BoundaryConditions? Seems like one then  could move the
# the  boundary data value val directly to BoundaryCondition
# Not sure how one would do this tho.
"""
    BoundaryData

A type for storing boundary data, e.g. constant, space-dependent, time-dependent etc.
Subtypes of `BoundaryData` should store the boundary data in a field `val`. The exception
to this is ZeroBoundaryData.
"""
abstract type BoundaryData end

"""
    ConstantBoundaryData

`val` is a scalar value of type T
"""
struct ConstantBoundaryData{T<:Number} <: BoundaryData
    val::T
end

"""
    SpaceDependentBoundaryData

`val` is a function of dimensionality equal to the boundary
"""
struct SpaceDependentBoundaryData{T<:Function} <: BoundaryData
    val::T
end

"""
    TimeDependentBoundaryData

`val` is a scalar function val(t)
"""
struct TimeDependentBoundaryData{T<:Function} <: BoundaryData
    val::T
end

"""
    SpaceTimeDependentBoundaryData

`val` is a timedependent function returning the spacedependent
    boundary data at a specific time. For instance, if f(t,x)
    is the function describing the spacetimedependent  boundary data then
    val(t*) returns the function g(x) = f(t*,x...)
"""
struct SpaceTimeDependentBoundaryData{T<:Function} <: BoundaryData
    val::T

    function SpaceTimeDependentBoundaryData(f::Function)
        val(t) = (args...) -> f(t,args...)
        return new{typeof(val)}(val)
    end
end

"""
    ZeroBoundaryData
"""
struct ZeroBoundaryData <: BoundaryData end


"""
    discretize(::BoundaryData, boundary_grid)

Returns an anonymous time-dependent function f, such that f(t) is
a `LazyArray` holding the `BoundaryData` discretized on `boundary_grid`.
"""
# TODO: Is the return type of discretize really a good interface
# for the boundary data?
# Moreover, instead of explicitly converting to a LazyArray here
# should we defer this to eval_on (and extend eval_on for scalars as well)?
# I.e. if eval_on returns a LazyArray, the boundary data is lazy. Otherwise
# it is preallocated.

function discretize(bd::ConstantBoundaryData, boundary_grid)
    return t -> LazyTensors.LazyConstantArray(bd.val, size(boundary_grid))
end

function discretize(bd::TimeDependentBoundaryData, boundary_grid)
    return t -> LazyTensors.LazyConstantArray(bd.val(t), size(boundary_grid))
end

function discretize(bd::SpaceDependentBoundaryData, boundary_grid)
    return t -> eval_on(boundary_grid, bd.val)
end

function discretize(bd::SpaceTimeDependentBoundaryData, boundary_grid)
    return t -> eval_on(boundary_grid, bd.val(t))
end

function discretize(::ZeroBoundaryData, boundary_grid)
    return t -> LazyTensors.LazyConstantArray(zero(eltype(boundary_grid)), size(boundary_grid))
end

"""
    BoundaryCondition

A type for implementing data needed in order to impose a boundary condition.
Subtypes refer to perticular types of boundary conditions, e.g. Neumann conditions.
"""
abstract type BoundaryCondition{T<:BoundaryData} end

"""
    data(::BoundaryCondition)

Returns the data stored by the `BoundaryCondition`.
"""
data(bc::BoundaryCondition) = bc.data
 

struct NeumannCondition{BD<:BoundaryData} <: BoundaryCondition{BD}
    data::BD
    id::BoundaryIdentifier 
end

struct DirichletCondition{BD<:BoundaryData} <: BoundaryCondition{BD}
    data::BD
    id::BoundaryIdentifier
end