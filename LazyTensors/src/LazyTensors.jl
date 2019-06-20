module LazyTensors


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



"""
    LazyTensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}

Struct for lazy transpose of a TensorMapping.

If a mapping implements the the `apply_transpose` method this allows working with
the transpose of mapping `m` by using `m'`. `m'` will work as a regular TensorMapping lazily calling
the appropriate methods of `m`.
"""
struct LazyTensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}
    tm::TensorMapping{T,R,D}
end
export LazyTensorMappingTranspose

# # TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
Base.adjoint(t::TensorMapping) = LazyTensorMappingTranspose(t)
Base.adjoint(t::LazyTensorMappingTranspose) = t.tm

apply(tm::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {T,R,D} = apply_transpose(tm.tm, v, I...)
apply_transpose(tm::LazyTensorMappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,D} = apply(tm.tm, v, I...)

range_size(tmt::LazyTensorMappingTranspose{T,R,D}, d_size::NTuple{R,Integer}) where {T,R,D} = domain_size(tmt.tm, domain_size)
domain_size(tmt::LazyTensorMappingTranspose{T,R,D}, r_size::NTuple{D,Integer}) where {T,R,D} = range_size(tmt.tm, range_size)


"""
    LazyTensorMappingApplication{T,R,D} <: AbstractArray{T,R}

Struct for lazy application of a TensorMapping. Created using `*`.

Allows the result of a `TensorMapping` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the LazyTensorMappingApplication object can be created by `m*v`.
The actual result will be calcualted when indexing into `m*v`.
"""
struct LazyTensorMappingApplication{T,R,D} <: AbstractArray{T,R}
    t::TensorMapping{T,R,D}
    o::AbstractArray{T,D}
end
export LazyTensorMappingApplication

Base.:*(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorMappingApplication(tm,o)

Base.getindex(ta::LazyTensorMappingApplication{T,R,D}, I::Vararg) where {T,R,D} = apply(ta.t, ta.o, I...)
Base.size(ta::LazyTensorMappingApplication{T,R,D}) where {T,R,D} = range_size(ta.t,size(ta.o))
# TODO: What else is needed to implement the AbstractArray interface?


# # We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
Base.:*(args::Union{TensorMapping{T}, AbstractArray{T}}...) where T = foldr(*,args)
# # Should we overload some other infix binary operator?
# →(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorMappingApplication(tm,o)
# TODO: We need to be really careful about good error messages.
# For example what happens if you try to multiply LazyTensorMappingApplication with a TensorMapping(wrong order)?



# struct LazyTensorMappingComposition{T,R,K,D} <: TensorMapping{T,R,D}
#     t1::TensorMapping{T,R,K}
#     t2::TensorMapping{T,K,D}
# end

# Base.:∘(s::TensorMapping{T,R,K}, t::TensorMapping{T,K,D}) where {T,R,K,D} = LazyTensorMappingComposition(s,t)

# function range_size(tm::LazyTensorMappingComposition{T,R,K,D}, domain_size::NTuple{D,Integer}) where {T,R,K,D}
#     range_size(tm.t1, domain_size(tm.t2, domain_size))
# end

# function domain_size(tm::LazyTensorMappingComposition{T,R,K,D}, range_size::NTuple{R,Integer}) where {T,R,K,D}
#     domain_size(tm.t1, domain_size(tm.t2, range_size))
# end

# function apply(c::LazyTensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
#     apply(c.t1, LazyTensorMappingApplication(c.t2,v), I...)
# end

# function apply_transpose(c::LazyTensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
#     apply_transpose(c.t2, LazyTensorMappingApplication(c.t1',v), I...)
# end

# # Have i gone too crazy with the type parameters? Maybe they aren't all needed?

# export →


end # module
