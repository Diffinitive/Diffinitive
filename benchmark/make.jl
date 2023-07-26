rev = nothing
baseline = nothing
target = nothing

if "--rev" ∈ ARGS
    i = findlast(==("--rev"), ARGS)
    rev = ARGS[i+1]
end

if "--target" ∈ ARGS
    i = findlast(==("--target"), ARGS)
    target = ARGS[i+1]
end

if "--baseline" ∈ ARGS
    i = findlast(==("--baseline"), ARGS)
    baseline = ARGS[i+1]
end

include("benchmark_utils.jl")
main(;rev, target, baseline)
