"""
    LazyTensor{R,D}

Describes a mapping of a `D` dimension tensor to an `R` dimension tensor.
The action of the mapping is implemented through the method
```julia
    apply(t::LazyTensor{R,D}, v::AbstractArray{<:Any,D}, I::Vararg) where {R,D}
```

The size of the range and domain that the operator works with should be returned by
the functions
```julia
    range_size(::LazyTensor)
    domain_size(::LazyTensor)
```
to allow querying for one or the other.
"""
abstract type LazyTensor{R,D} end

"""
    apply(t::LazyTensor{R,D}, v::AbstractArray{<:Any,D}, I::Vararg) where {R,D}

Return the result of the mapping for a given index.
"""
function apply end

"""
    range_dim(::LazyTensor)
Return the dimension of the range space of a given mapping
"""
range_dim(::LazyTensor{R,D}) where {R,D} = R

"""
    domain_dim(::LazyTensor)
Return the dimension of the domain space of a given mapping
"""
domain_dim(::LazyTensor{R,D}) where {R,D} = D


"""
    range_size(::LazyTensor)

Return the range size for the mapping.
"""
function range_size end

"""
    domain_size(::LazyTensor)

Return the domain size for the mapping.
"""
function domain_size end


"""
    Base.adjoint(::LazyTensor)

The adjoint of the mapping as a `LazyTensor`.
"""
Base.adjoint(::LazyTensor)


# TODO: Add docs for check functions

check_domain_size(::Type{Bool}, t::LazyTensor, sz) = domain_size(t) == sz

function check_domain_size(t::LazyTensor, sz)
    if !check_domain_size(Bool, t, sz)
        throw(DomainSizeMismatch(t, sz))
    end
end


check_range_size(::Type{Bool}, t::LazyTensor, sz) = range_size(t) == sz

function check_range_size(t::LazyTensor, sz)
    if !check_range_size(Bool, t, sz)
        throw(RangeSizeMismatch(t, sz))
    end
end


function check_equal_size(::Type{Bool}, t1::LazyTensor, t2::LazyTensor)
    return check_domain_size(Bool, t2, domain_size(t1)) & check_range_size(Bool, t2, range_size(t1))
end

function check_equal_size(::Type{Bool}, t1::LazyTensor, t2::LazyTensor, ts::LazyTensor...)
    return check_equal_size(Bool, t1, t2) & check_equal_size(Bool, t1, ts...)
end

function check_equal_size(t1::LazyTensor, t2::LazyTensor)
    if !check_equal_size(Bool, t1, t2)
        check_domain_size(t2, domain_size(t1))
        check_range_size(t2, range_size(t1))
    end
end

function check_equal_size(t1::LazyTensor, t2::LazyTensor, ts::LazyTensor...)
    check_equal_size(t1, t2)
    check_equal_size(t1, ts...)
end

check_composable(::Type{Bool}, t1::LazyTensor, t2::LazyTensor) = check_domain_size(Bool, t1, range_size(t2))

function check_composable(t1::LazyTensor, t2::LazyTensor)
    if !check_composable(Bool, t1, t2)
        throw(DomainSizeMismatch(t1, range_size(t2)))
    end
end


struct DomainSizeMismatch <: Exception
    t::LazyTensor
    sz
end

function Base.showerror(io::IO, err::DomainSizeMismatch)
    print(io, "DomainSizeMismatch: ")
    print(io, "domain size $(domain_size(err.t)) of LazyTensor not matching size $(err.sz)")
end


struct RangeSizeMismatch <: Exception
    t::LazyTensor
    sz
end

function Base.showerror(io::IO, err::RangeSizeMismatch)
    print(io, "RangeSizeMismatch: ")
    print(io, "range size $(range_size(err.t)) of LazyTensor not matching size $(err.sz)")
end
