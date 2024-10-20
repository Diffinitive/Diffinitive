using Pkg

function update_directory(d)
    Pkg.activate(d)
    Pkg.update()
    println()
end

update_directory(".")
update_directory("benchmark")
update_directory("docs")
update_directory("test")
