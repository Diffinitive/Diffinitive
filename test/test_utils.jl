"""
    cmp_fields(s1,s2)

Compares the fields of two structs s1, s2, using the == operator.
"""
function cmp_fields(s1::T,s2::T) where T
    f = fieldnames(T)
    return getfield.(Ref(s1),f) == getfield.(Ref(s2),f)
end
