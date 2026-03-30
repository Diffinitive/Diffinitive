"""
    componentview(gf, component_index...)

A view of `gf` with only the components specified by `component_index...`.

# Examples
```julia-repl
julia> componentview([[1,2], [2,3], [3,4]],2)
3-element ArrayComponentView{Int64, Vector{Int64}, 1, Vector{Vector{Int64}}, Tuple{Int64}}:
 2
 3
 4
```
"""
componentview(gf, component_index...) = ArrayComponentView(gf, component_index)

struct ArrayComponentView{CT,T,D,AT <: AbstractArray{T,D}, IT} <: AbstractArray{CT,D}
    v::AT
    component_index::IT

    function ArrayComponentView(v, component_index)
        CT = typeof(first(v)[component_index...])
        return new{CT, eltype(v), ndims(v), typeof(v), typeof(component_index)}(v,component_index)
    end
end

_array_type(v::ArrayComponentView) = _array_type(typeof(v))
_array_type(::Type{ArrayComponentView{CT,T,D,AT,IT}}) where {CT,T,D,AT,IT} = AT

Base.size(cv::ArrayComponentView) = size(cv.v)
Base.getindex(cv::ArrayComponentView, i::Int) = cv.v[i][cv.component_index...]
Base.getindex(cv::ArrayComponentView, I::Vararg{Int}) = cv.v[I...][cv.component_index...]
Base.IndexStyle(ACT::Type{<:ArrayComponentView}) = IndexStyle(_array_type(ACT))

# TODO: Implement `setindex!`?
# TODO: Implement a more general ComponentView that can handle non-AbstractArrays.
