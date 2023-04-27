if "--rev" ∈ ARGS
    i = findlast(==("--rev"), ARGS)
    args = parse(Int,ARGS[i+1])
elseif ("--target","--baseline") ∈ ARGS
    i = findlast(==("--target"), ARGS)
    j = findlast(==("--baseline"), ARGS)
    args = (ARGS[i+1],ARGS[j+1])
else
    args = ()
end

include("benchmark_utils.jl")
main(args...)