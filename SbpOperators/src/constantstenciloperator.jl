export apply

abstract type ConstantStencilOperator end

# Apply for different regions Lower/Interior/Upper or Unknown region
@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Lower})
    return @inbounds h*h*apply(op.closureStencils[Int(i)], v, Int(i))
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Interior})
    return @inbounds h*h*apply(op.innerStencil, v, Int(i))
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Index{Upper})
    N = length(v)
    return @inbounds h*h*Int(op.parity)*apply_backwards(op.closureStencils[N-Int(i)+1], v, Int(i))
end

@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, index::Index{Unknown})
    cSize = closuresize(op)
    N = length(v)

    i = Int(index)

    if 0 < i <= cSize
        return apply(op, h, v, Index{Lower}(i))
    elseif cSize < i <= N-cSize
        return apply(op, h, v, Index{Interior}(i))
    elseif N-cSize < i <= N
        return apply(op, h, v, Index{Upper}(i))
    else
        error("Bounds error") # TODO: Make this more standard
    end
end

# Wrapper functions for using regular indecies without specifying regions
@inline function apply(op::ConstantStencilOperator, h::Real, v::AbstractVector, i::Int)
    return apply(op, h, v, Index{Unknown}(i))
end
