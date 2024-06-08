"""
    TensorApplication{T,R,D} <: LazyArray{T,R}

Struct for lazy application of a LazyTensor. Created using `*`.

Allows the result of a `LazyTensor` applied to a vector to be treated as an `AbstractArray`.
With a mapping `m` and a vector `v` the TensorApplication object can be created by `m*v`.
The actual result will be calculated when indexing into `m*v`.
"""
struct TensorApplication{T,R,D, TM<:LazyTensor{<:Any,R,D}, AA<:AbstractArray{<:Any,D}} <: LazyArray{T,R}
    t::TM
    o::AA

    function TensorApplication(t::LazyTensor{<:Any,R,D}, o::AbstractArray{<:Any,D}) where {R,D}
        @boundscheck check_domain_size(t, size(o))
        I = ntuple(i->1, range_dim(t))
        T = typeof(apply(t,o,I...))
        return new{T,R,D,typeof(t), typeof(o)}(t,o)
    end
end

function Base.getindex(ta::TensorApplication{T,R}, I::Vararg{Any,R}) where {T,R}
    @boundscheck checkbounds(ta, Int.(I)...)
    return @inbounds apply(ta.t, ta.o, I...)
end
Base.@propagate_inbounds Base.getindex(ta::TensorApplication{T,1} where T, I::CartesianIndex{1}) = ta[Tuple(I)...] # Would otherwise be caught in the previous method.
Base.size(ta::TensorApplication) = range_size(ta.t)


"""
    TensorTranspose{T,R,D} <: LazyTensor{T,D,R}

Struct for lazy transpose of a LazyTensor.

If a mapping implements the the `apply_transpose` method this allows working with
the transpose of mapping `m` by using `m'`. `m'` will work as a regular LazyTensor lazily calling
the appropriate methods of `m`.
"""
struct TensorTranspose{T,R,D, TM<:LazyTensor{T,R,D}} <: LazyTensor{T,D,R}
    tm::TM
end

# # TBD: Should this be implemented on a type by type basis or through a trait to provide earlier errors?
# Jonatan 2020-09-25: Is the problem that you can take the transpose of any LazyTensor even if it doesn't implement `apply_transpose`?
Base.adjoint(tm::LazyTensor) = TensorTranspose(tm)
Base.adjoint(tmt::TensorTranspose) = tmt.tm

apply(tmt::TensorTranspose{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D} = apply_transpose(tmt.tm, v, I...)
apply_transpose(tmt::TensorTranspose{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmt.tm, v, I...)

range_size(tmt::TensorTranspose) = domain_size(tmt.tm)
domain_size(tmt::TensorTranspose) = range_size(tmt.tm)


struct ElementwiseTensorOperation{Op,T,R,D,T1<:LazyTensor{T,R,D},T2<:LazyTensor{T,R,D}} <: LazyTensor{T,R,D}
    tm1::T1
    tm2::T2

    function ElementwiseTensorOperation{Op,T,R,D}(tm1::T1,tm2::T2) where {Op,T,R,D, T1<:LazyTensor{T,R,D},T2<:LazyTensor{T,R,D}}
        @boundscheck check_domain_size(tm2, domain_size(tm1))
        @boundscheck check_range_size(tm2, range_size(tm1))
        return new{Op,T,R,D,T1,T2}(tm1,tm2)
    end
end

ElementwiseTensorOperation{Op}(s,t) where Op = ElementwiseTensorOperation{Op,eltype(s), range_dim(s), domain_dim(s)}(s,t)

apply(tmBinOp::ElementwiseTensorOperation{:+,T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) + apply(tmBinOp.tm2, v, I...)
apply(tmBinOp::ElementwiseTensorOperation{:-,T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D} = apply(tmBinOp.tm1, v, I...) - apply(tmBinOp.tm2, v, I...)

range_size(tmBinOp::ElementwiseTensorOperation) = range_size(tmBinOp.tm1)
domain_size(tmBinOp::ElementwiseTensorOperation) = domain_size(tmBinOp.tm1)


"""
    TensorComposition{T,R,K,D}

Lazily compose two `LazyTensor`s, so that they can be handled as a single `LazyTensor`.
"""
struct TensorComposition{T,R,K,D, TM1<:LazyTensor{T,R,K}, TM2<:LazyTensor{T,K,D}} <: LazyTensor{T,R,D}
    t1::TM1
    t2::TM2

    function TensorComposition(t1::LazyTensor{T,R,K}, t2::LazyTensor{T,K,D}) where {T,R,K,D}
        @boundscheck check_domain_size(t1, range_size(t2))
        return new{T,R,K,D, typeof(t1), typeof(t2)}(t1,t2)
    end
end

range_size(tm::TensorComposition) = range_size(tm.t1)
domain_size(tm::TensorComposition) = domain_size(tm.t2)

function apply(c::TensorComposition{T,R,K,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,K,D}
    apply(c.t1, c.t2*v, I...)
end

function apply_transpose(c::TensorComposition{T,R,K,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,K,D}
    apply_transpose(c.t2, c.t1'*v, I...)
end

"""
    TensorComposition(tm, tmi::IdentityTensor)
    TensorComposition(tmi::IdentityTensor, tm)

Composes a `LazyTensor` `tm` with an `IdentityTensor` `tmi`, by returning `tm`
"""
function TensorComposition(tm::LazyTensor{T,R,D}, tmi::IdentityTensor{T,D}) where {T,R,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tm
end

function TensorComposition(tmi::IdentityTensor{T,R}, tm::LazyTensor{T,R,D}) where {T,R,D}
    @boundscheck check_domain_size(tmi, range_size(tm))
    return tm
end
# Specialization for the case where tm is an IdentityTensor. Required to resolve ambiguity.
function TensorComposition(tm::IdentityTensor{T,D}, tmi::IdentityTensor{T,D}) where {T,D}
    @boundscheck check_domain_size(tm, range_size(tmi))
    return tmi
end

Base.:*(a::T, tm::LazyTensor{T}) where T = TensorComposition(ScalingTensor{T,range_dim(tm)}(a,range_size(tm)), tm)
Base.:*(tm::LazyTensor{T}, a::T) where T = a*tm
Base.:-(tm::LazyTensor) where T = (-one(eltype(tm)))*tm

"""
    InflatedTensor{T,R,D} <: LazyTensor{T,R,D}

An inflated `LazyTensor` with dimensions added before and after its actual dimensions.
"""
struct InflatedTensor{T,R,D,D_before,R_middle,D_middle,D_after, TM<:LazyTensor{T,R_middle,D_middle}} <: LazyTensor{T,R,D}
    before::IdentityTensor{T,D_before}
    tm::TM
    after::IdentityTensor{T,D_after}

    function InflatedTensor(before, tm::LazyTensor{T}, after) where T
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
    InflatedTensor(before, tm, after)
    InflatedTensor(before,tm)
    InflatedTensor(tm,after)

The outer product of `before`, `tm` and `after`, where `before` and `after` are `IdentityTensor`s.

If one of `before` or `after` is left out, a 0-dimensional `IdentityTensor` is used as the default value.

If `tm` already is an `InflatedTensor`, `before` and `after` will be extended instead of
creating a nested `InflatedTensor`.
"""
InflatedTensor(::IdentityTensor, ::LazyTensor, ::IdentityTensor)

function InflatedTensor(before, itm::InflatedTensor, after)
    return InflatedTensor(
        IdentityTensor(before.size...,  itm.before.size...),
        itm.tm,
        IdentityTensor(itm.after.size..., after.size...),
    )
end

InflatedTensor(before::IdentityTensor, tm::LazyTensor{T}) where T = InflatedTensor(before,tm,IdentityTensor{T}())
InflatedTensor(tm::LazyTensor{T}, after::IdentityTensor) where T = InflatedTensor(IdentityTensor{T}(),tm,after)
# Resolve ambiguity between the two previous methods
InflatedTensor(I1::IdentityTensor{T}, I2::IdentityTensor{T}) where T = InflatedTensor(I1,I2,IdentityTensor{T}())

# TODO: Implement some pretty printing in terms of ⊗. E.g InflatedTensor(I(3),B,I(2)) -> I(3)⊗B⊗I(2)

function range_size(itm::InflatedTensor)
    return concatenate_tuples(
        range_size(itm.before),
        range_size(itm.tm),
        range_size(itm.after),
    )
end

function domain_size(itm::InflatedTensor)
    return concatenate_tuples(
        domain_size(itm.before),
        domain_size(itm.tm),
        domain_size(itm.after),
    )
end

function apply(itm::InflatedTensor{T,R,D}, v::AbstractArray{<:Any,D}, I::Vararg{Any,R}) where {T,R,D}
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(dim_before, dim_domain, dim_range, dim_after, I...)

    v_inner = view(v, view_index...)
    return apply(itm.tm, v_inner, inner_index...)
end

function apply_transpose(itm::InflatedTensor{T,R,D}, v::AbstractArray{<:Any,R}, I::Vararg{Any,D}) where {T,R,D}
    dim_before = range_dim(itm.before)
    dim_domain = domain_dim(itm.tm)
    dim_range = range_dim(itm.tm)
    dim_after = range_dim(itm.after)

    view_index, inner_index = split_index(dim_before, dim_range, dim_domain, dim_after, I...)

    v_inner = view(v, view_index...)
    return apply_transpose(itm.tm, v_inner, inner_index...)
end


@doc raw"""
    LazyOuterProduct(tms...)

Creates a `TensorComposition` for the outer product of `tms...`.
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
    itm1 = InflatedTensor(tm1, IdentityTensor{T}(range_size(tm2)))
    itm2 = InflatedTensor(IdentityTensor{T}(domain_size(tm1)),tm2)

    return itm1∘itm2
end

LazyOuterProduct(t1::IdentityTensor{T}, t2::IdentityTensor{T}) where T = IdentityTensor{T}(t1.size...,t2.size...)
LazyOuterProduct(t1::LazyTensor, t2::IdentityTensor) = InflatedTensor(t1, t2)
LazyOuterProduct(t1::IdentityTensor, t2::LazyTensor) = InflatedTensor(t1, t2)

LazyOuterProduct(tms::Vararg{LazyTensor}) = foldl(LazyOuterProduct, tms)



"""
    inflate(tm::LazyTensor, sz, dir)

Inflate `tm` such that it gets the size `sz` in all directions except `dir`.
Here `sz[dir]` is ignored and replaced with the range and domains size of
`tm`.

An example of when this operation is useful is when extending a one
dimensional difference operator `D` to a 2D grid of a certain size. In that
case we could have

```julia
Dx = inflate(D, (10,10), 1)
Dy = inflate(D, (10,10), 2)
```
"""
function inflate(tm::LazyTensor, sz, dir)
    Is = IdentityTensor{eltype(tm)}.(sz)
    parts = Base.setindex(Is, tm, dir)
    return foldl(⊗, parts)
end

function check_domain_size(tm::LazyTensor, sz)
    if domain_size(tm) != sz
        throw(DomainSizeMismatch(tm,sz))
    end
end

function check_range_size(tm::LazyTensor, sz)
    if range_size(tm) != sz
        throw(RangeSizeMismatch(tm,sz))
    end
end

struct DomainSizeMismatch <: Exception
    tm::LazyTensor
    sz
end

function Base.showerror(io::IO, err::DomainSizeMismatch)
    print(io, "DomainSizeMismatch: ")
    print(io, "domain size $(domain_size(err.tm)) of LazyTensor not matching size $(err.sz)")
end


struct RangeSizeMismatch <: Exception
    tm::LazyTensor
    sz
end

function Base.showerror(io::IO, err::RangeSizeMismatch)
    print(io, "RangeSizeMismatch: ")
    print(io, "range size $(range_size(err.tm)) of LazyTensor not matching size $(err.sz)")
end
