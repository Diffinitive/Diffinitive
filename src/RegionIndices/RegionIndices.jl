module RegionIndices

abstract type Region end
struct Interior <: Region end
struct Lower    <: Region end
struct Upper    <: Region end
struct Unknown  <: Region end

export Region, Interior, Lower, Upper, Unknown

struct Index{R<:Region, T<:Integer}
    i::T

    Index{R,T}(i::T) where {R<:Region,T<:Integer} = new{R,T}(i)
    Index{R}(i::T) where {R<:Region,T<:Integer} = new{R,T}(i)
    Index(i::T, ::Type{R}) where {R<:Region,T<:Integer} = Index{R,T}(i)
    Index(t::Tuple{T, DataType}) where {R<:Region,T<:Integer} = Index{t[2],T}(t[1]) # TBD: This is not very specific in what types are allowed in t[2]. Can this be fixed?
end

export Index

# Index(R::Type{<:Region}) = Index{R}

## Vill kunna skriva
## IndexTupleType(Int, (Lower, Interior))
Index(R::Type{<:Region}, T::Type{<:Integer}) = Index{R,T}
IndexTupleType(T::Type{<:Integer},R::NTuple{N, DataType} where N) = Tuple{Index.(R, T)...}

Base.convert(::Type{T}, i::Index{R,T} where R) where T = i.i
Base.convert(::Type{CartesianIndex}, I::NTuple{N,Index} where N) = CartesianIndex(convert.(Int, I))

Base.Int(I::Index) = I.i
Base.to_index(I::Index) = Int(I) #How to get this to work for all cases??
Base.getindex(A::AbstractArray{T,N}, I::NTuple{N,Index}) where {T,N} = A[I...] #Is this ok??

function Index(i::Integer, boundary_width::Integer, dim_size::Integer)
    return Index{getregion(i,boundary_width,dim_size)}(i)
end

IndexTuple(t::Vararg{Tuple{T, DataType}}) where T<:Integer = Index.(t)
export IndexTuple

# TODO: Use the values of the region structs, e.g. Lower(), for the region parameter instead of the types.
# For example the following works:
#   (Lower(),Upper()) isa NTuple{2, Region} -> true
#   typeof((Lower(),Upper()))               -> Tuple{Lower,Upper}
function regionindices(gridsize::NTuple{Dim,Integer}, closuresize::Integer, region::NTuple{Dim,DataType}) where Dim
    return regionindices(gridsize, ntuple(x->closuresize,Dim), region)
end

function regionindices(gridsize::NTuple{Dim,Integer}, closuresize::NTuple{Dim,Integer}, region::NTuple{Dim,DataType}) where Dim
    regions = map(getrange,gridsize,closuresize,region)
    return CartesianIndices(regions)
end

export regionindices

function getregion(i::Integer, boundary_width::Integer, dim_size::Integer)
	if 0 < i <= boundary_width
        return Lower
    elseif boundary_width < i <= dim_size-boundary_width
        return Interior
    elseif dim_size-boundary_width < i <= dim_size
        return Upper
    else
        error("Bounds error") # TODO: Make this more standard
    end
end

export getregion

function getrange(gridsize::Integer, closuresize::Integer, region::DataType)
    if region == Lower
        r = 1:closuresize
    elseif region == Interior
        r = (closuresize+1):(gridsize - closuresize)
    elseif region == Upper
        r = (gridsize - closuresize + 1):gridsize
    else
        error("Unspecified region")
    end
    return r
end

end # module
