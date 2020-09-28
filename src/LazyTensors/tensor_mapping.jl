"""
    TensorMapping{T,R,D}

Describes a mapping of a D dimension tensor to an R dimension tensor.
The action of the mapping is implemented through the method

    apply(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}

The size of the range and domain that the operator works with should be returned by
the functions

    range_size(::TensorMapping)
    domain_size(::TensorMapping)

to allow querying for one or the other.

Optionally the action of the transpose may be defined through
    apply_transpose(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}
"""
abstract type TensorMapping{T,R,D} end
export TensorMapping

"""
    apply(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}

Return the result of the mapping for a given index.
"""
function apply end
export apply

"""
    apply_transpose(t::TensorMapping{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {R,D,T}

Return the result of the transposed mapping for a given index.
"""
function apply_transpose end
export apply_transpose

"""
Return the dimension of the range space of a given mapping
"""
range_dim(::TensorMapping{T,R,D}) where {T,R,D} = R

"""
Return the dimension of the domain space of a given mapping
"""
domain_dim(::TensorMapping{T,R,D}) where {T,R,D} = D

export range_dim, domain_dim

"""
    range_size(M::TensorMapping)

Return the range size for the mapping.
"""
function range_size end

"""
    domain_size(M::TensorMapping)

Return the domain size for the mapping.
"""
function domain_size end

export range_size, domain_size

# TODO: Think about boundschecking!
