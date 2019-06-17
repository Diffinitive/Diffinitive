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

range_size(tmt::TensorMappingTranspose{T,R,D}, domain_size::NTuple{D,Integer}) where {T,R,D} = domain_size(tmt.tm, domain_size)
domain_size(tmt::TensorMappingTranspose{T,R,D}, range_size::NTuple{D,Integer}) where {T,R,D} = range_size(tmt.tm, range_size)



struct TensorApplication{T,R,D} <: AbstractArray{T,R}
	t::TensorMapping{R,D}
	o::AbstractArray{T,D}
end

Base.size(ta::TensorApplication{T,R,D}) where {T,R,D} = range_size(ta.t,size(ta.o))
Base.getindex(ta::TensorApplication{T,R,D}, I::Vararg) where {T,R,D} = apply(ta.t, ta.o, I...)
# TODO: What else is needed to implement the AbstractArray interface?
import Base.*
→(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplication(tm,o)
# Should we overload some other infix binary operator?
# We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
*(args::Union{TensorMapping{T}, AbstractArray{T}}...) where T = foldr(*,args)
*(tm::TensorMapping{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplication(tm,o)
*(scalar, ta::TensorApplication{T,R,D}) where {T,R,D} = scalar*ta.o
*(ta::TensorApplication{T,R,D}, scalar::Number) where {T,R,D} = scalar*ta
# We need to be really careful about good error messages.
# For example what happens if you try to multiply TensorApplication with a TensorMapping(wrong order)?

# NOTE: TensorApplicationExpressions attempt to handle the situation when a TensorMapping
# acts on a TensorApplication +- AbstractArray, such that the expression still can be
# evaluated lazily per index.
# TODO: Better naming of both struct and members
# Since this is a lower layer which shouldnt be exposed, my opinion is that
# we can afford to be quite verbose.
struct TensorApplicationExpression{T,R,D} <: AbstractArray{T,R}
	ta::TensorApplication{R,D}
	o::AbstractArray{T,D}
end
Base.size(tae::TensorApplicationExpression) = size(tae.ta) #TODO: Not sure how to handle this
Base.getindex(tae::TensorApplicationExpression, I::Vararg) = tae.ta[I...] + tae.o[I...]
import Base.+
import Base.-
+(ta::TensorApplication{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = TensorApplicationExpression(ta,o)
+(o::AbstractArray{T,D},ta::TensorApplication{T,R,D}) where {T,R,D} = ta + o
-(ta::TensorApplication{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = ta + -o
-(o::AbstractArray{T,D},ta::TensorApplication{T,R,D}) where {T,R,D} = -ta + o

# NOTE: Another (quite neat) way to handle lazy evaluation of
# TensorApplication + AbstractArray is by using broadcasting.
# However, with the drafted implementation below a
# TensorApplication+AbstractArray now results in a generic function and we would
# then need to define TensorMapping*generic function which does not seem like a
# good idea.
# NOTE: Could one use MappedArrays.jl instead?
#
# # Lazy evaluations of expressions on TensorApplications
# # TODO: Need to decide on some good naming here.
# +(ta::TensorApplication,o::AbstractArray) = I -> ta[I] + o[I]
# +(o::AbstractArray,ta::TensorApplication) = ta+o
# *(scalar::Number,ta::TensorApplication) = I -> scalar*ta[I]
# *(ta::TensorApplication,scalar::Number) = scalar*ta
# -(ta::TensorApplication,o::AbstractArray) = ta + -o
# -(o::AbstractArray + ta::TensorApplication) = -ta + o

struct TensorMappingComposition{T,R,K,D} <: TensorMapping{T,R,D} where K<:typeof(R)
	t1::TensorMapping{T,R,K}
	t2::TensorMapping{T,K,D}
end

import Base.∘
∘(s::TensorMapping{T,R,K}, t::TensorMapping{T,K,D}) where {T,R,K,D} = TensorMappingComposition(s,t)

function range_size(tm::TensorMappingComposition{T,R,K,D}, domain_size::NTuple{D,Integer}) where {T,R,K,D}
	range_size(tm.t1, domain_size(tm.t2, domain_size))
end

function domain_size(tm::TensorMappingComposition{T,R,K,D}, range_size::NTuple{R,Integer}) where {T,R,K,D}
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
