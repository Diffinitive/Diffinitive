"""
    TensorMapping{T,R,D}

Describes a mapping of a D dimension tensor to an R dimension tensor.
The action of the mapping is implemented through the method

    apply(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}

The size of range tensor should be dependent on the size of the domain tensor
and the type should implement the methods

    range_size(::TensorMapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D}
    domain_size(::TensorMapping{T,R,D}, range_size::NTuple{R,Integer}) where {T,R,D}

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
    range_size(M::TensorMapping, domain_size)

Return the resulting range size for the mapping applied to a given domain_size
"""
function range_size end

"""
    domain_size(M::TensorMapping, range_size)

Return the resulting domain size for the mapping applied to a given range_size
"""
function domain_size end

export range_size, domain_size
# TODO: Think about boundschecking!


"""
    TensorOperator{T,D}

A `TensorMapping{T,D,D}` where the range and domain tensor have the same number of
dimensions and the same size.
"""
abstract type TensorOperator{T,D} <: TensorMapping{T,D,D} end
export TensorOperator
domain_size(::TensorOperator{T,D}, range_size::NTuple{D,Integer}) where {T,D} = range_size
range_size(::TensorOperator{T,D}, domain_size::NTuple{D,Integer}) where {T,D} = domain_size
