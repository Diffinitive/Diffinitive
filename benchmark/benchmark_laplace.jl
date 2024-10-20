using Diffinitive
using Diffinitive.SbpOperators
using Diffinitive.Grids
using Diffinitive.RegionIndices
using BenchmarkTools

# TODO: Move the below benchmarks into the benchmark suite

const operator_path = sbp_operators_path()*"standard_diagonal.toml"

function benchmark_const_coeff_1d(;N = 100, order = 4)
    stencil_set = read_stencil_set(operator_path; order=order)
    g = equidistant_grid(0., 1., N)
    D = second_derivative(g, stencil_set)
    u = rand(size(g)...)
    u_xx = rand(size(g)...)
        
    b_naive = @benchmark $u_xx .= $D*$u
    b_reg = @benchmark $apply_region_1d!($u_xx,$u,$D)
    b_thrd = @benchmark $apply_region_threaded_1d!($u_xx,$u,$D)
    print_benchmark_result("Naive apply",b_naive)
    print_benchmark_result("Region apply",b_reg)
    print_benchmark_result("Threaded region apply",b_thrd)
end

function benchmark_var_coeff_1d(;N = 100, order = 4)
    stencil_set = read_stencil_set(operator_path; order=order)
    g = equidistant_grid(0., 1., N)
    c = rand(size(g)...)
    c_lz = eval_on(g, x -> 0.5)
    D = second_derivative_variable(g, c, stencil_set)
    D_lz = second_derivative_variable(g, c_lz, stencil_set)
    u = rand(size(g)...)
    u_xx = rand(size(g)...)
    
    b_naive = @benchmark $u_xx .= $D*$u
    b_naive_lz = @benchmark $u_xx .= $D_lz*$u
    b_reg = @benchmark $apply_region_1d!($u_xx,$u,$D)
    b_reg_lz = @benchmark $apply_region_1d!($u_xx,$u,$D_lz)
    b_thrd = @benchmark $apply_region_threaded_1d!($u_xx,$u,$D)
    b_thrd_lz = @benchmark $apply_region_threaded_1d!($u_xx,$u,$D_lz)
    print_benchmark_result("Naive apply",b_naive)
    print_benchmark_result("Naive apply lazy coeff",b_naive_lz)
    print_benchmark_result("Region apply",b_reg)
    print_benchmark_result("Region apply lazy coeff",b_reg_lz)
    print_benchmark_result("Threaded region apply",b_thrd)
    print_benchmark_result("Threaded region apply lazy coeff",b_thrd_lz)
end

function benchmark_const_coeff_2d(;N = 100, order = 4)
    stencil_set = read_stencil_set(operator_path; order=order)
    g = equidistant_grid((0.,0.,),(1.,1.), N, N)
    D = Laplace(g, stencil_set)
    u = rand(size(g)...)
    u_xx = rand(size(g)...)
    if order == 2
        clz_sz = 1
    elseif order == 4
        clz_sz = 4
    else
        error()
    end

    b_naive = @benchmark $u_xx .= $D*$u
    b_reg = @benchmark $apply_region_2d!($u_xx,$u,$D,$clz_sz)
    b_thrd = @benchmark $apply_region_threaded_2d!($u_xx,$u,$D,$clz_sz)
    print_benchmark_result("Naive apply",b_naive)
    print_benchmark_result("Region apply",b_reg)
    print_benchmark_result("Threaded region apply",b_thrd)
end

function benchmark_var_coeff_2d(;N = 100, order = 4)
    stencil_set = read_stencil_set(operator_path; order=order)
    g = equidistant_grid((0.,0.,),(1.,1.), N, N)
    c = rand(size(g)...)
    c_lz = eval_on(g, x-> 0.5)
    D = second_derivative_variable(g, c, stencil_set, 1) + second_derivative_variable(g, c, stencil_set, 2)
    D_lz = second_derivative_variable(g, c_lz, stencil_set, 1) + second_derivative_variable(g, c_lz, stencil_set, 2)
    u = rand(size(g)...)
    u_xx = rand(size(g)...)

    if order == 2
        clz_sz = 1
    elseif order == 4
        clz_sz = 6
    else
        error()
    end
    
    # Check correctnesss
    # u_xx .= D*u
    # u_xx_tmp = zeros(size(u_xx)...)
    # u_xx_tmp .= u_xx
    # apply_region_threaded_2d!(u_xx, u, D, clz_sz)

    # @show sum(abs.(u_xx_tmp .- u_xx))
    # @show pointer(u_xx_tmp) == pointer(u_xxs

    
    b_naive = @benchmark $u_xx .= $D*$u
    b_naive_lz = @benchmark $u_xx .= $D_lz*$u
    b_reg = @benchmark $apply_region_2d!($u_xx,$u,$D, $clz_sz)
    b_reg_lz = @benchmark $apply_region_2d!($u_xx,$u,$D_lz, $clz_sz)
    b_thrd = @benchmark $apply_region_threaded_2d!($u_xx,$u,$D, $clz_sz)
    b_thrd_lz = @benchmark $apply_region_threaded_2d!($u_xx,$u,$D_lz, $clz_sz)
    print_benchmark_result("Naive apply",b_naive)
    print_benchmark_result("Naive apply lazy coeff",b_naive_lz)
    print_benchmark_result("Region apply",b_reg)
    print_benchmark_result("Region apply lazy coeff",b_reg_lz)
    print_benchmark_result("Threaded region apply",b_thrd)
    print_benchmark_result("Threaded region apply lazy coeff",b_thrd_lz)
end

function print_benchmark_result(title_str,res)
    if title_str[1] != ' '
        title_str = lpad(title_str,length(title_str)+1, " ")
    end
    if title_str[end] != ' '
        title_str = rpad(title_str,length(title_str)+1, " ")
    end
    tot_len = 76
    pad_len = Int(tot_len/2)
    header = lpad(title_str,pad_len,"*")
    header = rpad(header,tot_len,"*")
    bottom = repeat("*",tot_len)
    println(header)
    display(res)
    println(bottom)
    return
end

function apply_region_1d!(u_xx, u, D)
    clz_sz = SbpOperators.closure_size(D)
    tm = D*u
    for i ∈ @view eachindex(u)[1:clz_sz]
        u_xx[i] = tm[Index{Lower}(i)]
    end
    for i ∈ @view eachindex(u)[clz_sz+1:end-clz_sz]
        u_xx[i] = tm[Index{Interior}(i)]
    end
    for i ∈ @view eachindex(u)[end-clz_sz+1:end]
        u_xx[i] = tm[Index{Upper}(i)]
    end
end

function apply_region_threaded_1d!(u_xx, u, D)
    clz_sz = SbpOperators.closure_size(D)
    tm = D*u
    for i ∈ @view eachindex(u)[1:clz_sz]
        u_xx[i] = tm[Index{Lower}(i)]
    end
    Threads.@threads for i ∈ @view eachindex(u)[clz_sz+1:end-clz_sz]
        u_xx[i] = tm[Index{Interior}(i)]
    end
    for i ∈ @view eachindex(u)[end-clz_sz+1:end]
        u_xx[i] = tm[Index{Upper}(i)]
    end
end

function apply_region_2d!(u_xx, u, D, clz_sz)
    tm = D*u
    for I ∈ @view CartesianIndices(u)[1:clz_sz,1:clz_sz]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Lower}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[1:clz_sz,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[1:clz_sz,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Upper}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,1:clz_sz]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Lower}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Upper}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,1:clz_sz]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Lower}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Upper}(I[2])]
    end
end

function apply_region_threaded_2d!(u_xx, u, D, clz_sz)
    tm = D*u
    for I ∈ @view CartesianIndices(u)[1:clz_sz,1:clz_sz]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Lower}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[1:clz_sz,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[1:clz_sz,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Lower}(I[1]),Index{Upper}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,1:clz_sz]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Lower}(I[2])]
    end
    Threads.@threads for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[clz_sz+1:end-clz_sz,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Interior}(I[1]),Index{Upper}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,1:clz_sz]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Lower}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,clz_sz+1:end-clz_sz]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Interior}(I[2])]
    end
    for I ∈ @view CartesianIndices(u)[end-clz_sz+1:end,end-clz_sz+1:end]
        u_xx[I] = tm[Index{Upper}(I[1]),Index{Upper}(I[2])]
    end
end
