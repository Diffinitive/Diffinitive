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
