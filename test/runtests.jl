using Diffinitive
using Test
using JET
using Aqua
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

function run_testfiles(path, globs)
    for name ∈ readdir(path)
        filepath = joinpath(path, name)

        if isdir(filepath)
            @testset "$name" begin
                run_testfiles(filepath, globs)
            end
        end

        if endswith(name, "_test.jl") && any(occursin.(globs, filepath))
            log_and_time(filepath) do
                @testset "$name" begin
                    include(filepath)
                end
            end
        end
    end
end

function log_and_time(f, msg)
    printstyled("Running "; bold=true, color=:green)
    print(msg)

    t_start = time()
    f()
    t_end = time()
    Δt = t_end - t_start
    printstyled(" ($(round(Δt, digits=2)) s)"; color=:light_black)
    println()
end

testsetname = isempty(ARGS) ? "Diffinitive.jl" : "["*join(ARGS, ", ")*"]"

@testset "$testsetname" begin
    if isempty(ARGS)
        log_and_time("code quality tests using Aqua.jl") do
            @testset "Code quality (Aqua.jl)" begin
                Aqua.test_all(Diffinitive)
            end
        end

        log_and_time("code linting using JET.jl") do
            @testset "Code linting (JET.jl)" begin
                JET.test_package(Diffinitive; target_defined_modules = true)
            end
        end
    end

    run_testfiles(ARGS)
    println()
end
