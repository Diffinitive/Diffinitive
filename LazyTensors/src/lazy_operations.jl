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



# TODO: Write tests and documentation for LazyTensorMappingComposition
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
