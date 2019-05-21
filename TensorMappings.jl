module TensorMappings
# Needs a better name ImplicitTensorMappings? Get rid of "Tensor" in the name_

abstract type TensorMapping{T,R,D} end

range_dim(::TensorMapping{T,R,D}) where {T,R,D} = R
domain_dim(::TensorMapping{T,R,D}) where {T,R,D} = D
# range_size(::TensorMapping{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D}
# domain_size(::TensorMapping{T,R,D}, range_size::NTuple{R,Integer}) where {T,R,D}

# apply(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T} =
# apply_transpose(t::TensorMapping{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {R,D,T} =
# Implementing apply_transpose and domain_size is only needed if you want to take transposes of the TensorMapping.
# TODO: Think about boundschecking!

abstract type TensorOperator{T,D} <: TensorMapping{T,D,D} end
domain_size(::TensorOperator{T,D}, range_size::NTuple{D,Integer}) where {T,D} = range_size
range_size(::TensorOperator{T,D}, domain_size::NTuple{D,Integer}) where {T,D} = domain_size



# Allow using the ' operator:
struct TensorMappingTranspose{T,R,D} <: TensorMapping{T,D,R}
	tm::TensorMapping{T,R,D}
end

Base.adjoint(t::TensorMapping) = TensorMappingTranspose(t)
# TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
Base.adjoint(t::TensorMappingTranspose) = t.tm

apply(tm::TensorMappingTranspose{T,R,D}, v::AbstractArray{T,R}, I::Vararg) where {T,R,D} = apply_transpose(tm.tm, v, I...)
apply_transpose(tm::TensorMappingTranspose{T,R,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,D} = apply(tm.tm, v, I...)

range_size(tmt::TensorMappingTranspose{T,R,D}, domain_size::NTuple{D,Integer}) = domain_size(tmt.tm, domain_size)
domain_size(tmt::TensorMappingTranspose{T,R,D}, range_size::NTuple{D,Integer}) = range_size(tmt.tm, range_size)



struct TensorApplication{T,R,D} <: AbstractArray{T,R}
	t::TensorMapping{R,D}
	o::AbstractArray{T,D}
end

Base.size(ta::TensorApplication) = range_size(ta.t,size(ta.o))
Base.getindex(tm::TensorApplication, I::Vararg) = apply(tm.t, tm.o, I...)
# TODO: What else is needed to implement the AbstractArray interface?

→(t::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplication(t,o)
# Should we overload some other infix binary operator?
# We need the associativity to be a→b→c = a→(b→c), which is the case for '→'

import Base.*
*(args::Union{TensorMapping{T}, AbstractArray{T}}...) where T = foldr(*,args)
*(t::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplication(t,o)
# We need to be really careful about good error messages.
# For example what happens if you try to multiply TensorApplication with a TensorMapping(wrong order)?



struct TensorMappingComposition{T,R,K,D} <: TensorMapping{T,R,D}
	t1::TensorMapping{T,R,K}
	t2::TensorMapping{T,K,D}
end

import Base.∘
∘(s::TensorMapping{T,R,K}, t::TensorMapping{T,K,D}) where {T,R,K,D} = TensorMappingComposition(s,t)

function range_size(tm::TensorMappingComposition{T,R,K,D}, domain_size::NTuple{D,Integer}) where {T,R,D}
	range_size(tm.t1, domain_size(tm.t2, domain_size))
end

function domain_size(tm::TensorMappingComposition{T,R,K,D}, range_size::NTuple{R,Integer}) where {T,R,D}
	domain_size(tm.t1, domain_size(tm.t2, range_size))
end

function apply(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
	apply(c.t1, TensorApplication(c.t2,v), I...)
end

function apply_transpose(c::TensorMappingComposition{T,R,K,D}, v::AbstractArray{T,D}, I::Vararg) where {T,R,K,D}
	apply_transpose(c.t2, TensorApplication(c.t1',v), I...)
end

# Have i gone too crazy with the type parameters? Maybe they aren't all needed?


export apply
export apply_transpose
export range_dim
export domain_dim
export range_size
export →


# # Automatic dimension expansion?
# struct TensorOperator1dAd2d{T,I} <: TensorOperator{T,2}
# 	t::TensorOperator{T,1}
# end

end #module
