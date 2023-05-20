using Test
using Glob

"""
    run_testfiles()
    run_testfiles(path, globs)

Find and run all files with filenames ending with "_test.jl". If `path` is omitted the test folder is assumed.
The argument `globs` can optionally be supplied to filter which test files are run.
"""
function run_testfiles(args)
    if isempty(args)
        globs = [fn"./*"]
    else
        globs = Glob.FilenameMatch.("./".*args)
    end

    run_testfiles(".", globs)
end

function  run_testfiles(path, globs)
    for name ∈ readdir(path)
        filepath = joinpath(path, name)

        if isdir(filepath)
            @testset "$name" begin
                run_testfiles(filepath, globs)
            end
        end

        if endswith(name, "_test.jl") && any(occursin.(globs, filepath))
            printstyled("Running "; bold=true, color=:green)
            print(filepath)

            t_start = time()
            @testset "$name" begin
                include(filepath)
            end
            t_end = time()

            Δt = t_end - t_start
            printstyled(" ($(round(Δt, digits=2)) s)"; color=:light_black)
            println()
        end
    end
end

testsetname = isempty(ARGS) ? "Sbplib.jl" : "["*join(ARGS, ", ")*"]"

@testset "$testsetname" begin
    run_testfiles(ARGS)
    println()
end
