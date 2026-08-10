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


# TODO: Add tests for check functions
# TODO: Add docs for check functions

function check_domain_size(t::LazyTensor, sz)
    if domain_size(t) != sz
        throw(DomainSizeMismatch(t, sz))
    end
end

function check_range_size(t::LazyTensor, sz)
    if range_size(t) != sz
        throw(RangeSizeMismatch(t, sz))
    end
end

function check_equal_size(ts::Vararg{LazyTensor})
    map(ts) do t
        check_domain_size(t, domain_size(ts[1]))
        check_range_size(t, range_size(ts[1]))
    end
end

function check_composable(t1::LazyTensor, t2::LazyTensor)
    check_domain_size(t1, range_size(t2))
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
