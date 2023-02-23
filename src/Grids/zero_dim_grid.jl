struct ZeroDimGrid{T,S} <: Grid{T,0}
    p::S

    function ZeroDimGrid(p)
        T = eltype(p)
        S = typeof(p)
        return new{T,S}(p)
    end
end

Base.size(g::ZeroDimGrid) = ()
Base.getindex(g::ZeroDimGrid) = g.p
Base.eachindex(g::ZeroDimGrid) = CartesianIndices(())

# Indexing interface
# TODO
# Iteration interface
# TODO
