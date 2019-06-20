module LazyTensor


"""
    Mapping{T,R,D}

Describes a mapping of a D dimension tensor to an R dimension tensor.
The action of the mapping is implemented through the method

    apply(t::Mapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}

The size of range tensor should be dependent on the size of the domain tensor
and the type should implement the methods

    range_size(::Mapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D}
    domain_size(::Mapping{T,R,D}, range_size::NTuple{R,Integer}) where {T,R,D}

to allow querying for one or the other.

Optionally the action of the transpose may be defined through
    apply_transpose(t::Mapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}
"""
abstract type Mapping{T,R,D} end

"""
    apply(t::Mapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T}

Return the result of the mapping for a given index.
"""
function apply end
export apply

"""
    apply_transpose(t::Mapping{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {R,D,T}

Return the result of the transposed mapping for a given index.
"""
function apply_transpose end
export apply_transpose

"""
Return the dimension of the range space of a given mapping
"""
range_dim(::Mapping{T,R,D}) where {T,R,D} = R

"""
Return the dimension of the domain space of a given mapping
"""
domain_dim(::Mapping{T,R,D}) where {T,R,D} = D

export range_dim, domain_dim

"""
    range_size(M::Mapping, domain_size)

Return the resulting range size for the mapping applied to a given domain_size
"""
function range_size end

"""
    domain_size(M::Mapping, range_size)

Return the resulting domain size for the mapping applied to a given range_size
"""
function domain_size end

export range_size, domain_size
# TODO: Think about boundschecking!


"""
    Operator{T,D}

A `Mapping{T,D,D}` where the range and domain tensor have the same number of
dimensions and the same size.
"""
abstract type Operator{T,D} <: Mapping{T,D,D} end
domain_size(::Operator{T,D}, range_size::NTuple{D,Integer}) where {T,D} = range_size
range_size(::Operator{T,D}, domain_size::NTuple{D,Integer}) where {T,D} = domain_size



"""
    MappingTranspose{T,R,D} <: Mapping{T,D,R}

Struct for lazy transpose of a Mapping.

If a mapping implements the the `apply_transpose` method this allows working with
the transpose of mapping `m` by using `m'`. `m'` will work as a regular Mapping lazily calling
the appropriate methods of `m`.
"""
struct MappingTranspose{T,R,D} <: Mapping{T,D,R}
    tm::Mapping{T,R,D}
end

# # TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
Base.adjoint(t::Mapping) = MappingTranspose(t)
Base.adjoint(t::MappingTranspose) = t.tm

apply(tm::MappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {T,R,D} = apply_transpose(tm.tm, v, I...)
apply_transpose(tm::MappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,D} = apply(tm.tm, v, I...)

range_size(tmt::MappingTranspose{T,R,D}, d_size::NTuple{R,Integer}) where {T,R,D} = domain_size(tmt.tm, domain_size)
domain_size(tmt::MappingTranspose{T,R,D}, r_size::NTuple{D,Integer}) where {T,R,D} = range_size(tmt.tm, range_size)


"""
    Application{T,R,D} <: AbstractArray{T,R}

Struct for lazy application of a Mapping. Created using `*`.

Allows the result of a `Mapping` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the Application object can be created by `m*v`.
The actual result will be calcualted when indexing into `m*v`.
"""
struct Application{T,R,D} <: AbstractArray{T,R}
    t::Mapping{T,R,D}
    o::AbstractArray{T,D}
end

Base.:*(tm::Mapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = Application(tm,o)

Base.getindex(ta::Application{T,R,D}, I::Vararg) where {T,R,D} = apply(ta.t, ta.o, I...)
Base.size(ta::Application{T,R,D}) where {T,R,D} = range_size(ta.t,size(ta.o))
# TODO: What else is needed to implement the AbstractArray interface?


# # We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
Base.:*(args::Union{Mapping{T}, AbstractArray{T}}...) where T = foldr(*,args)
# # Should we overload some other infix binary operator?
# →(tm::Mapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = Application(tm,o)
# TODO: We need to be really careful about good error messages.
# For example what happens if you try to multiply Application with a Mapping(wrong order)?



# struct TensorMappingComposition{T,R,K,D} <: Mapping{T,R,D}
#     t1::Mapping{T,R,K}
#     t2::Mapping{T,K,D}
# end

# Base.:∘(s::Mapping{T,R,K}, t::Mapping{T,K,D}) where {T,R,K,D} = TensorMappingComposition(s,t)

# function range_size(tm::TensorMappingComposition{T,R,K,D}, domain_size::NTuple{D,Integer}) where {T,R,K,D}
#     range_size(tm.t1, domain_size(tm.t2, domain_size))
# end

# function domain_size(tm::TensorMappingComposition{T,R,K,D}, range_size::NTuple{R,Integer}) where {T,R,K,D}
#     domain_size(tm.t1, domain_size(tm.t2, range_size))
# end

# function apply(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
#     apply(c.t1, Application(c.t2,v), I...)
# end

# function apply_transpose(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
#     apply_transpose(c.t2, Application(c.t1',v), I...)
# end

# # Have i gone too crazy with the type parameters? Maybe they aren't all needed?

# export →


end # module
