struct ZeroDimGrid{T,S,RD} <: Grid{T,0,RD}
    p::S

    function ZeroDimGrid(p)
        T = eltype(p)
        S = typeof(p)
        RD = length(p)
        return new{T,S,RD}(p)
    end
end

Base.size(g::ZeroDimGrid) = ()
Base.getindex(g::ZeroDimGrid) = g.p
Base.eachindex(g::ZeroDimGrid) = CartesianIndices(())
