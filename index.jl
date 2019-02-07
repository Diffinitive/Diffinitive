abstract type Region end
struct Interior <: Region end
struct Lower    <: Region end
struct Upper    <: Region end
struct Unknown  <: Region end

struct Index{R<:Region, T<:Integer}
    i::T

    Index{R,T}(i::T) where {R<:Region,T<:Integer} = new{R,T}(i)
    Index{R}(i::T) where {R<:Region,T<:Integer} = new{R,T}(i)
    Index(i::T, ::Type{R}) where {R<:Region,T<:Integer} = Index{R,T}(i)
    Index(t::Tuple{T, DataType}) where {R<:Region,T<:Integer} = Index{t[2],T}(t[1]) # TBD: This is not very specific in what types are allowed in t[2]. Can this be fixed?
end

# Index(R::Type{<:Region}) = Index{R}

## Vill kunna skriva
## IndexTupleType(Int, (Lower, Interior))
Index(R::Type{<:Region}, T::Type{<:Integer}) = Index{R,T}
IndexTupleType(T::Type{<:Integer},R::NTuple{N, DataType} where N) = Tuple{Index.(R, T)...}

Base.convert(::Type{T}, i::Index{R,T} where R) where T = i.i
Base.convert(::Type{CartesianIndex}, I::NTuple{N,Index} where N) = CartesianIndex(convert.(Int, I))

Base.Int(I::Index) = I.i

function Index(i::Integer, boundary_width::Integer, dim_size::Integer)
    if 0 < i <= boundary_width
        return Index{Lower}(i)
    elseif boundary_width < i <= dim_size-boundary_width
        return Index{Interior}(i)
    elseif dim_size-boundary_width < i <= dim_size
        return Index{Upper}(i)
    else
        error("Bounds error") # TODO: Make this more standard
    end
end

IndexTuple(t::Vararg{Tuple{T, DataType}}) where T<:Integer = Index.(t)

function regionindices(gridsize::NTuple{Dim,Integer}, closuresize::Integer, region::NTuple{Dim,DataType}) where Dim
    return regionindices(gridsize, ntuple(x->closuresize,Dim), region)
end

function regionindices(gridsize::NTuple{Dim,Integer}, closuresize::NTuple{Dim,Integer}, region::NTuple{Dim,DataType}) where Dim
    regions = map(getunitrange,gridsize,closuresize,region)
    return CartesianIndices(regions)
end

function getunitrange(gridsize::Integer, closuresize::Integer, region::R) where R
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
