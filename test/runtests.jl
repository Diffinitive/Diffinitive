using Test
using Glob

"""
    run_testfiles()
    run_testfiles(path)
    run_testfiles(path, glob)

Find and run all files with filenames ending with "_test.jl". If `path` is omitted the test folder is assumed.
The argument `glob` can optionally be supplied to filter which test files are run.
"""
function run_testfiles(args)
    if isempty(args)
        glob = fn"./*"
    else
        glob = Glob.FilenameMatch("./"*args[1]) #TBD: Allow multiple filters?
    end

    run_testfiles(".", glob)
end

function  run_testfiles(path, glob)
    for name ∈ readdir(path)
        filepath = joinpath(path, name)

        if isdir(filepath)
            @testset "$name" begin
                run_testfiles(filepath, glob)
            end
        end

        if endswith(name, "_test.jl") && occursin(glob, filepath)
            printstyled("Running "; bold=true, color=:green)
            println(filepath)
            @testset "$name" begin
                include(filepath)
            end
        end
    end
end

testsetname = isempty(ARGS) ? "Sbplib.jl" : ARGS[1]

@testset "$testsetname" begin
    run_testfiles(ARGS)
end
