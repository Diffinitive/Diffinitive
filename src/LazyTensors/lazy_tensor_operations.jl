# TBD: Is there a good way to split this file?

"""
    LazyTensorApplication{T,R,D} <: LazyArray{T,R}

Struct for lazy application of a LazyTensor. Created using `*`.

Allows the result of a `LazyTensor` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the LazyTensorApplication object can be created by `m*v`.
The actual result will be calcualted when indexing into `m*v`.
"""
struct LazyTensorApplication{T,R,D, TM<:LazyTensor{<:Any,R,D}, AA<:AbstractArray{<:Any,D}} <: LazyArray{T,R}
    t::TM
    o::AA

    function LazyTensorApplication(t::LazyTensor{<:Any,R,D}, o::AbstractArray{<:Any,D}) where {R,D}
        I = ntuple(i->1, range_dim(t))
        T = typeof(apply(t,o,I...))
        return new{T,R,D,typeof(t), typeof(o)}(t,o)
    end
end
# TODO: Do boundschecking on creation!

Base.getindex(ta::LazyTensorApplication{T,R}, I::Vararg{Any,R}) where {T,R} = apply(ta.t, ta.o, I...)
Base.getindex(ta::LazyTensorApplication{T,1}, I::CartesianIndex{1}) where {T} = apply(ta.t, ta.o, I.I...) # Would otherwise be caught in the previous method.
Base.size(ta::LazyTensorApplication) = range_size(ta.t)
# TODO: What else is needed to implement the AbstractArray interface?

Base.:*(a::LazyTensor, v::AbstractArray) = LazyTensorApplication(a,v)
Base.:*(a::LazyTensor, b::LazyTensor) = throw(MethodError(Base.:*,(a,b)))
Base.:*(a::LazyTensor, args::Union{LazyTensor, AbstractArray}...) = foldr(*,(a,args...))

# # We need the associativity to be a→b→c = a→(b→c), which is the case for '→'
# # Should we overload some other infix binary opesrator?
# →(tm::LazyTensor{T,R,D}, o::AbstractArray{T,D}) where {T,R,D} = LazyTensorApplication(tm,o)
# TODO: We need to be really careful about good error messages.
# For example what happens if you try to multiply LazyTensorApplication with a LazyTensor(wrong order)?

"""
    LazyTensorTranspose{T,R,D} <: LazyTensor{T,D,R}

Struct for lazy transpose of a LazyTensor.

If a mapping implements the the `apply_transpose` method this allows working with
the transpose of mapping `m` by using `m'`. `m'` will work as a regular LazyTensor lazily calling
the appropriate methods of `m`.
"""
struct LazyTensorTranspose{T,R,D, TM<:LazyTensor{T,R,D}} <: LazyTensor{T,D,R}
    tm::TM
end

# # TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
# Jonatan 2020-09-25: Is the problem that you can take the transpose of any LazyTensor even if it doesn't implement `apply_transpose`?
Base.adjoint(tm::LazyTensor) = LazyTensorTranspose(tm)
Base.adjoint(tmt::LazyTensorTranspose) = tmt.tm

apply(tmt::LazyTensorTranspose{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D} = apply_transpose(tmt.tm, v, I...)
apply_transpose(tmt::LazyTensorTranspose{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmt.tm, v, I...)

range_size(tmt::LazyTensorTranspose) = domain_size(tmt.tm)
domain_size(tmt::LazyTensorTranspose) = range_size(tmt.tm)


struct LazyLazyTensorBinaryOperation{Op,T,R,D,T1<:LazyTensor{T,R,D},T2<:LazyTensor{T,R,D}} <: LazyTensor{T,D,R}
    tm1::T1
    tm2::T2

    @inline function LazyLazyTensorBinaryOperation{Op,T,R,D}(tm1::T1,tm2::T2) where {Op,T,R,D, T1<:LazyTensor{T,R,D},T2<:LazyTensor{T,R,D}}
        return new{Op,T,R,D,T1,T2}(tm1,tm2)
    end
end
# TODO: Boundschecking in constructor.

apply(tmBinOp::LazyLazyTensorBinaryOperation{:+,T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) + apply(tmBinOp.tm2, v, I...)
apply(tmBinOp::LazyLazyTensorBinaryOperation{:-,T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) - apply(tmBinOp.tm2, v, I...)

range_size(tmBinOp::LazyLazyTensorBinaryOperation) = range_size(tmBinOp.tm1)
domain_size(tmBinOp::LazyLazyTensorBinaryOperation) = domain_size(tmBinOp.tm1)

Base.:+(tm1::LazyTensor{T,R,D}, tm2::LazyTensor{T,R,D}) where {T,R,D} = LazyLazyTensorBinaryOperation{:+,T,R,D}(tm1,tm2)
Base.:-(tm1::LazyTensor{T,R,D}, tm2::LazyTensor{T,R,D}) where {T,R,D} = LazyLazyTensorBinaryOperation{:-,T,R,D}(tm1,tm2)

"""
    LazyTensorComposition{T,R,K,D}

Lazily compose two `LazyTensor`s, so that they can be handled as a single `LazyTensor`.
"""
struct LazyTensorComposition{T,R,K,D, TM1<:LazyTensor{T,R,K}, TM2<:LazyTensor{T,K,D}} <: LazyTensor{T,R,D}
    t1::TM1
    t2::TM2

    @inline function LazyTensorComposition(t1::LazyTensor{T,R,K}, t2::LazyTensor{T,K,D}) where {T,R,K,D}
        @boundscheck check_domain_size(t1, range_size(t2))
        return new{T,R,K,D, typeof(t1), typeof(t2)}(t1,t2)
    end
end

range_size(tm::LazyTensorComposition) = range_size(tm.t1)
domain_size(tm::LazyTensorComposition) = domain_size(tm.t2)

function apply(c::LazyTensorComposition{T,R,K,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,K,D}
    apply(c.t1, c.t2*v, I...)
end

function apply_transpose(c::LazyTensorComposition{T,R,K,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,K,D}
    apply_transpose(c.t2, c.t1'*v, I...)
end

Base.@propagate_inbounds Base.:∘(s::LazyTensor, t::LazyTensor) = LazyTensorComposition(s,t)

"""
    LazyLinearMap{T,R,D,...}(A, range_indicies, domain_indicies)

LazyTensor defined by the AbstractArray A. `range_indicies` and `domain_indicies` define which indicies of A should
be considerd the range and domain of the LazyTensor. Each set of indices must be ordered in ascending order.

For instance, if A is a m x n matrix, and range_size = (1,), domain_size = (2,), then the LazyLinearMap performs the
standard matrix-vector product on vectors of size n.
"""
struct LazyLinearMap{T,R,D, RD, AA<:AbstractArray{T,RD}} <: LazyTensor{T,R,D}
    A::AA
    range_indicies::NTuple{R,Int}
    domain_indicies::NTuple{D,Int}

    function LazyLinearMap(A::AA, range_indicies::NTuple{R,Int}, domain_indicies::NTuple{D,Int}) where {T,R,D, RD, AA<:AbstractArray{T,RD}}
        if !issorted(range_indicies) || !issorted(domain_indicies)
            throw(DomainError("range_indicies and domain_indicies must be sorted in ascending order"))
        end

        return new{T,R,D,RD,AA}(A,range_indicies,domain_indicies)
    end
end

range_size(llm::LazyLinearMap) = size(llm.A)[[llm.range_indicies...]]
domain_size(llm::LazyLinearMap) = size(llm.A)[[llm.domain_indicies...]]

function apply(llm::LazyLinearMap{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D}
    view_index = ntuple(i->:,ndims(llm.A))
    for i ∈ 1:R
        view_index = Base.setindex(view_index, Int(I[i]), llm.range_indicies[i])
    end
    A_view = @view llm.A[view_index...]
    return sum(A_view.*v)
end

function apply_transpose(llm::LazyLinearMap{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D}
    apply(LazyLinearMap(llm.A, llm.domain_indicies, llm.range_indicies), v, I...)
end


"""
    IdentityTensor{T,D} <: LazyTensor{T,D,D}

The lazy identity LazyTensor for a given size. Usefull for building up higher dimensional tensor mappings from lower
dimensional ones through outer products. Also used in the Implementation for InflatedLazyTensor.
"""
struct IdentityTensor{T,D} <: LazyTensor{T,D,D}
    size::NTuple{D,Int}
end

IdentityTensor{T}(size::NTuple{D,Int}) where {T,D} = IdentityTensor{T,D}(size)
IdentityTensor{T}(size::Vararg{Int,D}) where {T,D} = IdentityTensor{T,D}(size)
IdentityTensor(size::Vararg{Int,D}) where D = IdentityTensor{Float64,D}(size)

range_size(tmi::IdentityTensor) = tmi.size
domain_size(tmi::IdentityTensor) = tmi.size

apply(tmi::IdentityTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = v[I...]
apply_transpose(tmi::IdentityTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = v[I...]

"""
    LazyTensorComposition(tm, tmi::IdentityTensor)
    LazyTensorComposition(tmi::IdentityTensor, tm)

Composes a `Tensormapping` `tm` with an `IdentityTensor` `tmi`, by returning `tm`
"""
function LazyTensorComposition(tm::LazyTensor{T,R,D}, tmi::IdentityTensor{T,D}) where {T,R,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tm
end

function LazyTensorComposition(tmi::IdentityTensor{T,R}, tm::LazyTensor{T,R,D}) where {T,R,D}
    @boundscheck check_domain_size(tmi, range_size(tm))
    return tm
end
# Specialization for the case where tm is an IdentityTensor. Required to resolve ambiguity.
function LazyTensorComposition(tm::IdentityTensor{T,D}, tmi::IdentityTensor{T,D}) where {T,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tmi
end
# TODO: Move the operator definitions to one place

"""
    ScalingTensor{T,D} <: LazyTensor{T,D,D}

A lazy tensor that scales its input with `λ`.
"""
struct ScalingTensor{T,D} <: LazyTensor{T,D,D}
    λ::T
    size::NTuple{D,Int}
end

LazyTensors.apply(tm::ScalingTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.λ*v[I...]
LazyTensors.apply_transpose(tm::ScalingTensor{T,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,D}) where {T,D} = tm.λ*v[I...]

LazyTensors.range_size(m::ScalingTensor) = m.size
LazyTensors.domain_size(m::ScalingTensor) = m.size

"""
    InflatedLazyTensor{T,R,D} <: LazyTensor{T,R,D}

An inflated `LazyTensor` with dimensions added before and afer its actual dimensions.
"""
struct InflatedLazyTensor{T,R,D,D_before,R_middle,D_middle,D_after, TM<:LazyTensor{T,R_middle,D_middle}} <: LazyTensor{T,R,D}
    before::IdentityTensor{T,D_before}
    tm::TM
    after::IdentityTensor{T,D_after}

    function InflatedLazyTensor(before, tm::LazyTensor{T}, after) where T
        R_before = range_dim(before)
        R_middle = range_dim(tm)
        R_after = range_dim(after)
        R = R_before+R_middle+R_after

        D_before = domain_dim(before)
        D_middle = domain_dim(tm)
        D_after = domain_dim(after)
        D = D_before+D_middle+D_after
        return new{T,R,D,D_before,R_middle,D_middle,D_after, typeof(tm)}(before, tm, after)
    end
end
"""
    InflatedLazyTensor(before, tm, after)
    InflatedLazyTensor(before,tm)
    InflatedLazyTensor(tm,after)

The outer product of `before`, `tm` and `after`, where `before` and `after` are `IdentityTensor`s.

If one of `before` or `after` is left out, a 0-dimensional `IdentityTensor` is used as the default value.

If `tm` already is an `InflatedLazyTensor`, `before` and `after` will be extended instead of
creating a nested `InflatedLazyTensor`.
"""
InflatedLazyTensor(::IdentityTensor, ::LazyTensor, ::IdentityTensor)

function InflatedLazyTensor(before, itm::InflatedLazyTensor, after)
    return InflatedLazyTensor(
        IdentityTensor(before.size...,  itm.before.size...),
        itm.tm,
        IdentityTensor(itm.after.size..., after.size...),
    )
end

InflatedLazyTensor(before::IdentityTensor, tm::LazyTensor{T}) where T = InflatedLazyTensor(before,tm,IdentityTensor{T}())
InflatedLazyTensor(tm::LazyTensor{T}, after::IdentityTensor) where T = InflatedLazyTensor(IdentityTensor{T}(),tm,after)
# Resolve ambiguity between the two previous methods
InflatedLazyTensor(I1::IdentityTensor{T}, I2::IdentityTensor{T}) where T = InflatedLazyTensor(I1,I2,IdentityTensor{T}())

# TODO: Implement some pretty printing in terms of ⊗. E.g InflatedLazyTensor(I(3),B,I(2)) -> I(3)⊗B⊗I(2)

function range_size(itm::InflatedLazyTensor)
    return flatten_tuple(
        range_size(itm.before),
        range_size(itm.tm),
        range_size(itm.after),
    )
end

function domain_size(itm::InflatedLazyTensor)
    return flatten_tuple(
        domain_size(itm.before),
        domain_size(itm.tm),
        domain_size(itm.after),
    )
end

function apply(itm::InflatedLazyTensor{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D}
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(Val(dim_before), Val(dim_domain), Val(dim_range), Val(dim_after), I...)

    v_inner = view(v, view_index...)
    return apply(itm.tm, v_inner, inner_index...)
end

function apply_transpose(itm::InflatedLazyTensor{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D}
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(Val(dim_before), Val(dim_range), Val(dim_domain), Val(dim_after), I...)

    v_inner = view(v, view_index...)
    return apply_transpose(itm.tm, v_inner, inner_index...)
end


@doc raw"""
    LazyOuterProduct(tms...)

Creates a `LazyTensorComposition` for the outerproduct of `tms...`.
This is done by separating the outer product into regular products of outer products involving only identity mappings and one non-identity mapping.

First let
```math
\begin{aligned}
A &= A_{I,J} \\
B &= B_{M,N} \\
C &= C_{P,Q} \\
\end{aligned}
```

where ``I``, ``M``, ``P`` are  multi-indexes for the ranges of ``A``, ``B``, ``C``, and ``J``, ``N``, ``Q`` are multi-indexes of the domains.

We use ``⊗`` to denote the outer product
```math
(A⊗B)_{IM,JN} = A_{I,J}B_{M,N}
```

We note that
```math
A⊗B⊗C = (A⊗B⊗C)_{IMP,JNQ} = A_{I,J}B_{M,N}C_{P,Q}
```
And that
```math
A⊗B⊗C = (A⊗I_{|M|}⊗I_{|P|})(I_{|J|}⊗B⊗I_{|P|})(I_{|J|}⊗I_{|N|}⊗C)
```
where ``|⋅|`` of a multi-index is a vector of sizes for each dimension. ``I_v`` denotes the identity tensor of size ``v[i]`` in each direction
To apply ``A⊗B⊗C`` we evaluate

```math
(A⊗B⊗C)v = [(A⊗I_{|M|}⊗I_{|P|})  [(I_{|J|}⊗B⊗I_{|P|}) [(I_{|J|}⊗I_{|N|}⊗C)v]]]
```
"""
function LazyOuterProduct end

function LazyOuterProduct(tm1::LazyTensor{T}, tm2::LazyTensor{T}) where T
    itm1 = InflatedLazyTensor(tm1, IdentityTensor{T}(range_size(tm2)))
    itm2 = InflatedLazyTensor(IdentityTensor{T}(domain_size(tm1)),tm2)

    return itm1∘itm2
end

LazyOuterProduct(t1::IdentityTensor{T}, t2::IdentityTensor{T}) where T = IdentityTensor{T}(t1.size...,t2.size...)
LazyOuterProduct(t1::LazyTensor, t2::IdentityTensor) = InflatedLazyTensor(t1, t2)
LazyOuterProduct(t1::IdentityTensor, t2::LazyTensor) = InflatedLazyTensor(t1, t2)

LazyOuterProduct(tms::Vararg{LazyTensor}) = foldl(LazyOuterProduct, tms)

⊗(a::LazyTensor, b::LazyTensor) = LazyOuterProduct(a,b)


function check_domain_size(tm::LazyTensor, sz)
    if domain_size(tm) != sz
        throw(SizeMismatch(tm,sz))
    end
end

struct SizeMismatch <: Exception
    tm::LazyTensor
    sz
end

function Base.showerror(io::IO, err::SizeMismatch)
    print(io, "SizeMismatch: ")
    print(io, "domain size $(domain_size(err.tm)) of LazyTensor not matching size $(err.sz)")
end
