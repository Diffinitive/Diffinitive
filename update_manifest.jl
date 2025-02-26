using Pkg

function update_and_log(root, d)
    printstyled("Updating '$d'"; color=:cyan, bold=true)
    println()
    dp = joinpath(root, d)

    update_directory(dp)
    println()
end

function update_directory(d)
    Pkg.activate(d)
    Pkg.update()
end

root = @__DIR__
update_and_log(root, ".")
update_and_log(root, "benchmark")
update_and_log(root, "docs")
update_and_log(root, "test")

