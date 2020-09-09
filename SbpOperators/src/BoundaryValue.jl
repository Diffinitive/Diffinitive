"""
    BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}
    eClosure::Stencil{T,M}
    bId::CartesianBoundary
end
export BoundaryValue

# TODO: This is obviouly strange. Is domain_size just discarded? Is there a way to avoid storing grid in BoundaryValue?
# Can we give special treatment to TensorMappings that go to a higher dim?
function LazyTensors.range_size(e::BoundaryValue{T}, domain_size::NTuple{1,Integer}) where T
    if dim(e.bId) == 1
        return (UnknownDim, domain_size[1])
    elseif dim(e.bId) == 2
        return (domain_size[1], UnknownDim)
    end
end
LazyTensors.domain_size(e::BoundaryValue{T}, range_size::NTuple{2,Integer}) where T = (range_size[3-dim(e.bId)],)
# TODO: Make a nicer solution for 3-dim(e.bId)

# TODO: Make this independent of dimension
function LazyTensors.apply(e::BoundaryValue{T}, v::AbstractArray{T}, I::NTuple{2,Index}) where T
    i = I[dim(e.bId)]
    j = I[3-dim(e.bId)]
    N_i = size(e.grid)[dim(e.bId)]
    return apply_boundary_value(e.op, v[j], i, N_i, region(e.bId))
end

function LazyTensors.apply_transpose(e::BoundaryValue{T}, v::AbstractArray{T}, I::NTuple{1,Index}) where T
    u = selectdim(v,3-dim(e.bId),Int(I[1]))
    return apply_boundary_value_transpose(e.op, u, region(e.bId))
end

function apply_boundary_value_transpose(op::ConstantStencilOperator, v::AbstractVector, ::Type{Lower})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply_stencil(op.eClosure,v,1)
end

function apply_boundary_value_transpose(op::ConstantStencilOperator, v::AbstractVector, ::Type{Upper})
    @boundscheck if length(v) < closuresize(op)
        throw(BoundsError())
    end
    apply_stencil_backwards(op.eClosure,v,length(v))
end
export apply_boundary_value_transpose

function apply_boundary_value(op::ConstantStencilOperator, v::Number, i::Index, N::Integer, ::Type{Lower})
    @boundscheck if !(0<length(Int(i)) <= N)
        throw(BoundsError())
    end
    op.eClosure[Int(i)-1]*v
end

function apply_boundary_value(op::ConstantStencilOperator, v::Number, i::Index,  N::Integer, ::Type{Upper})
    @boundscheck if !(0<length(Int(i)) <= N)
        throw(BoundsError())
    end
    op.eClosure[N-Int(i)]*v
end
export apply_boundary_value


"""
    BoundaryValue{T,N,M,K} <: TensorMapping{T,2,1}

Implements the boundary operator `e` as a TensorMapping
"""
struct BoundaryValue{D,T,M,R} <: TensorMapping{T,D,1}
    e:BoundaryOperator{T,M,R}
    bId::CartesianBoundary
end

function LazyTensors.apply_transpose(bv::BoundaryValue{T,M,Lower}, v::AbstractVector{T}, i::Index) where T
    u = selectdim(v,3-dim(bv.bId),Int(I[1]))
    return apply_transpose(bv.e, u, I)
end


"""
    BoundaryOperator{T,N,R} <: TensorMapping{T,1,1}

Implements the boundary operator `e` as a TensorMapping
"""
export BoundaryOperator
struct BoundaryOperator{T,M,R<:Region} <: TensorMapping{T,1,1}
    closure::Stencil{T,M}
end

function LazyTensors.range_size(e::BoundaryOperator, domain_size::NTuple{1,Integer})
    return UnknownDim
end

LazyTensors.domain_size(e::BoundaryOperator{T}, range_size::NTuple{1,Integer}) where T = range_size

function LazyTensors.apply_transpose(e::BoundaryOperator{T,M,Lower}, v::AbstractVector{T}, i::Index{Lower}) where T
    @boundscheck if length(v) < closuresize(e) #TODO: Use domain_size here?
        throw(BoundsError())
    end
    apply_stencil(e.closure,v,Int(i))
end

function LazyTensors.apply_transpose(e::BoundaryOperator{T,M,Upper}}, v::AbstractVector{T}, i::Index{Upper}) where T
    @boundscheck if length(v) < closuresize(e) #TODO: Use domain_size here?
        throw(BoundsError())
    end
    apply_stencil_backwards(e.closure,v,Int(i))
end

function LazyTensors.apply_transpose(e::BoundaryOperator{T}, v::AbstractVector{T}, i::Index) where T
    @boundscheck if length(v) < closuresize(e) #TODO: Use domain_size here?
        throw(BoundsError())
    end
    return eltype(v)(0)
end

#TODO: Implement apply in a meaningful way. Should it return a vector or a single value (perferable?) Should fit into  the
function LazyTensors.apply(e::BoundaryOperator, v::AbstractVector, i::Index)
    @boundscheck if !(0<length(Int(i)) <= length(v))
        throw(BoundsError())
    end
    return e.closure[Int(i)].*v
end
