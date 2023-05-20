"""
    ConstantInteriorScalingOperator{T,N} <: LazyTensor{T,1,1}

A one-dimensional operator scaling a vector. The first and last `N` points are
scaled with individual weights while all interior points are scaled using the
same factor.
"""
struct ConstantInteriorScalingOperator{T,N} <: LazyTensor{T,1,1}
    interior_weight::T
    closure_weights::NTuple{N,T}
    size::Int

    function ConstantInteriorScalingOperator(interior_weight::T, closure_weights::NTuple{N,T}, size::Int) where {T,N}
        if size < 2*length(closure_weights)
            throw(DomainError(size, "size must be larger that two times the closure size."))
        end

        return new{T,N}(interior_weight, closure_weights, size)
    end
end

function ConstantInteriorScalingOperator(grid::EquidistantGrid, interior_weight, closure_weights)
    return ConstantInteriorScalingOperator(interior_weight, Tuple(closure_weights), size(grid)[1])
end

closure_size(::ConstantInteriorScalingOperator{T,N}) where {T,N} = N

LazyTensors.range_size(op::ConstantInteriorScalingOperator) = (op.size,)
LazyTensors.domain_size(op::ConstantInteriorScalingOperator) = (op.size,)

# TBD: @inbounds in apply methods?
function LazyTensors.apply(op::ConstantInteriorScalingOperator, v::AbstractVector, i::Index{Lower})
    return op.closure_weights[Int(i)]*v[Int(i)]
end

function LazyTensors.apply(op::ConstantInteriorScalingOperator, v::AbstractVector, i::Index{Interior})
    return op.interior_weight*v[Int(i)]
end

function LazyTensors.apply(op::ConstantInteriorScalingOperator, v::AbstractVector, i::Index{Upper})
    return op.closure_weights[op.size[1]-Int(i)+1]*v[Int(i)]
end

function LazyTensors.apply(op::ConstantInteriorScalingOperator, v::AbstractVector, i)
    r = getregion(i, closure_size(op), op.size[1])
    return LazyTensors.apply(op, v, Index(i, r))
end

LazyTensors.apply_transpose(op::ConstantInteriorScalingOperator, v, i) = apply(op, v, i)
