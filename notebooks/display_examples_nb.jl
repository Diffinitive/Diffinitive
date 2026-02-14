### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 1f8a7cfa-94cc-41bf-a8c8-2dc5218741e0
begin
	using Diffinitive
	using Diffinitive.Grids
	using Diffinitive.LazyTensors
	using Diffinitive.SbpOperators
	using PlutoUI
	using StaticArrays
end

# ╔═╡ 885c60d7-d33c-4741-ae49-6a57510ec7b5
md"""
# Display tests
"""

# ╔═╡ 9ee3372a-e78d-4f74-84ce-e04208d1558d
repl_show(v) = repr(MIME("text/plain"), v) |> println

# ╔═╡ 51c02ced-f684-417f-83f1-cade4edda43f
md"""
Common julia objects to compare with:
"""

# ╔═╡ 25c90528-22cd-41ca-8572-ccd946928318
1 |> repl_show

# ╔═╡ e7f3e466-9833-428c-99ad-20bc9d88d951
[1,1] |> repl_show

# ╔═╡ 2e74f9b5-5b4f-4887-8a30-4655d560a45c
[1;; 2;;] |> repl_show

# ╔═╡ 365524b5-3182-4691-9817-1bbec1492c14
[1; 2;;] |> repl_show

# ╔═╡ d5725e1b-bc4f-4a95-975d-179c193908c9
[1; 2;; 3; 4;;] |> repl_show

# ╔═╡ b824ef8d-5026-4861-9a23-45a7939fd38c
"hej" |> repl_show

# ╔═╡ fbe365a2-f95e-4297-8326-c18d22932869
Dict("A" => 1, "B"=> 2) |> repl_show

# ╔═╡ 56670aff-0343-41cb-a653-35a61376dda4
1//2 |> repl_show

# ╔═╡ b5a6491e-a93e-4058-8ceb-be1dc4d4c100
BigInt(30) |> repl_show

# ╔═╡ 828d57a1-ee58-4204-8050-78127821a4c6
1:10 |> repl_show

# ╔═╡ 127d34f6-69f7-4082-a74b-0be86942f153
range(0,1,10) |> repl_show

# ╔═╡ c46a278e-a102-4544-82d8-7df816440410
rand(2,2,2,2) |> repl_show

# ╔═╡ 5aa7079c-8005-47f1-bb82-c35f3aa54b42
md"""
## Parameter spaces
"""

# ╔═╡ 08f493ed-189c-43f3-86f2-95fc475ec0e7
Interval(1,2) |> repl_show

# ╔═╡ f7244bf7-8266-469f-b07f-30c203d9af48
md"""
## Grids
"""

# ╔═╡ 0e14bd28-5dd1-44c4-abf4-23b70546bd49
equidistant_grid(0,1,11) |> repl_show

# ╔═╡ fcb74341-6b03-4ada-8f5d-bc245c23679b
equidistant_grid((0,0),(1,1),10,20) |> repl_show

# ╔═╡ 8dec053b-eaae-463d-800b-b8d89d5d550b
ZeroDimGrid(@SVector[1,2]) |> repl_show

# ╔═╡ c1172a36-c5d7-47dc-bc79-af0d43a8f6ee
let
	x̄((ξ, η)) = @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
    J((ξ, η)) = @SMatrix[
        2       1-2η;
        (2+η)*ξ 3+1/2*ξ^2;
    ]

	mapped_grid(x̄, J, 10,10) |> repl_show
end

# ╔═╡ 9c889176-865b-402d-81b5-71957d2878f7
md"""
## LazyArrays
"""

# ╔═╡ 85e8e748-e575-4a29-80c7-22d110578343
LazyTensors.LazyConstantArray(10, (5,)) |> repl_show

# ╔═╡ 804ad722-9081-4d1d-b0d2-c536a26fe20d
LazyTensors.LazyFunctionArray((i,j)->10*i+j, (3,4)) |> repl_show

# ╔═╡ a70c689d-0851-497f-938a-e5c92ce59ddb
md"""
## LazyTensors
"""

# ╔═╡ 2afde3fe-96ed-4d7e-a79b-fc880e0da268
LazyTensors.IdentityTensor(5) |> repl_show

# ╔═╡ 5451a071-14ae-47ae-99c5-4d65508d280f
LazyTensors.IdentityTensor(4,3) |> repl_show

# ╔═╡ b6b06fe9-de16-41ca-ad45-eef6dd038485
LazyTensors.ScalingTensor(2., (4,3)) |> repl_show

# ╔═╡ d1c8c3a0-76ec-4c32-853e-0471d71e5cf0
LazyTensors.DiagonalTensor([1,2,3,4]) |> repl_show

# ╔═╡ 6051c144-9982-4bd9-92f9-d0aaf3961872
LazyTensors.DenseTensor(rand(2,2,2,2), (1,2), (3,4)) |> repl_show

# ╔═╡ 12a9f430-f96b-43f2-bf63-149b5a028fd7
begin
	g1 = equidistant_grid(0,1,10)
	g2 = equidistant_grid((0,0),(1,1),10, 11)
	x̄((ξ, η)) = @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
    J((ξ, η)) = @SMatrix[
        2       1-2η;
        (2+η)*ξ 3+1/2*ξ^2;
    ]
	mg = mapped_grid(x̄, J, 10,10) 
	stencil_set2 = stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
	stencil_set4 = stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
end;

# ╔═╡ e44f8d91-1cbf-44be-bef1-40e60c4a777f
first_derivative(g1, stencil_set2) |> repl_show

# ╔═╡ c378b88a-1d74-44a2-bdc1-b371da478de8
first_derivative(g1, stencil_set4) |> repl_show

# ╔═╡ 9614cda7-b48c-4925-89b5-113cf514f20f
first_derivative(g2, stencil_set2, 1) |> repl_show

# ╔═╡ 725e9430-4821-4939-bfef-6c186d2dc500
first_derivative(g2, stencil_set4, 2) |> repl_show

# ╔═╡ e8ca54a1-a6db-40a1-b44a-73e175894df4
second_derivative(g1, stencil_set2) |> repl_show

# ╔═╡ 5d1f10fe-f620-469c-822c-55955a5541ad
second_derivative(g1, stencil_set4) |> repl_show

# ╔═╡ fdc531d4-9d27-41dc-ba79-6febefde223a
second_derivative(g2, stencil_set2, 1) |> repl_show

# ╔═╡ fff2e04a-357f-4254-996e-d5ccd9ff31f8
second_derivative(g2, stencil_set4, 2) |> repl_show

# ╔═╡ 959b071e-1ef6-4f29-aa8b-d88bfef80c00
second_derivative_variable(g1, map(x->2x, g1), stencil_set2) |> repl_show

# ╔═╡ 0a0f7e77-a789-4fb3-a4c2-7853b67788ec
second_derivative_variable(g1, map(x->2x, g1), stencil_set4) |> repl_show

# ╔═╡ 177e0893-fbb1-4bb5-a108-e5990e943ab7
 second_derivative_variable(g2, map(x->x[1]+x[2], g2), stencil_set2, 1) |> repl_show

# ╔═╡ f1b6bd54-baf6-4360-aec0-bd8d52497894
second_derivative_variable(g2, map(x->x[1]+x[2], g2), stencil_set4, 2) |> repl_show

# ╔═╡ a8f8343e-cd6f-451d-a2b3-e5f0f561f8af
undivided_skewed04(g1,4)[1] |> repl_show

# ╔═╡ 50291998-7194-4a0d-9c19-eacc48b3f5da
undivided_skewed04(g1,4)[2] |> repl_show

# ╔═╡ 1720af08-85e4-4502-b37e-9fa73008e221
undivided_skewed04(g2,4,1)[1] |> repl_show

# ╔═╡ 4a72f217-2423-4bc1-8452-eb28dde36689
undivided_skewed04(g2,4,2)[2] |> repl_show

# ╔═╡ 48a28ade-73bb-461d-ab96-82f92ed199c8
inner_product(g1, stencil_set2) |> repl_show

# ╔═╡ 3c681a63-94a6-4677-aef7-df903c463896
inner_product(g1, stencil_set4) |> repl_show

# ╔═╡ db735370-153a-40f9-b77f-9f60e30a35c4
inner_product(g2, stencil_set2) |> repl_show

# ╔═╡ 613ebac7-50bc-424c-8fa2-064b64c93319
inner_product(g2, stencil_set4) |> repl_show

# ╔═╡ 9e7d7667-960e-491f-8b83-e3b01a0db5b0
inverse_inner_product(g2, stencil_set4) |> repl_show

# ╔═╡ c3005e74-5b96-4b0c-9c57-b7d02968ed94
boundary_restriction(g1, stencil_set, LowerBoundary()) |> repl_show

# ╔═╡ fd019973-0d54-4e31-b43b-f53a704cb01c
boundary_restriction(g1, stencil_set, UpperBoundary()) |> repl_show

# ╔═╡ df2e8af0-9ca5-4972-8771-f8bea1591f85
boundary_restriction(g2, stencil_set, CartesianBoundary{1,LowerBoundary}()) |> repl_show

# ╔═╡ 8e538109-16a1-4af7-a986-9dc1455b7de7
boundary_restriction(g2, stencil_set, CartesianBoundary{2,UpperBoundary}()) |> repl_show

# ╔═╡ 5f3744a4-72d8-4448-820f-a928bfaaf825
normal_derivative(g1, stencil_set, LowerBoundary()) |> repl_show

# ╔═╡ 4529b5c4-4905-4fd3-9aaa-5f88faa841c8
normal_derivative(g1, stencil_set, UpperBoundary()) |> repl_show

# ╔═╡ c3bb0450-a7d5-44c1-9ca9-9e1ecf2db9f8
normal_derivative(g2, stencil_set, CartesianBoundary{1,LowerBoundary}()) |> repl_show

# ╔═╡ da63e57a-0794-4cd8-9941-ada0e5c1c40e
normal_derivative(g2, stencil_set, CartesianBoundary{2,UpperBoundary}()) |> repl_show

# ╔═╡ 0b951425-979c-4ac9-8581-f690b729bab4
md"""
## Tensor operations
"""

# ╔═╡ a39bb6e2-f1fe-4206-8ee3-88ff0c075233
begin
	Dx = first_derivative(g2, stencil_set2, 1)
	Dy = first_derivative(g2, stencil_set4, 2)

	v = map(x->sin(x[1]^2+x[2]^2), g2)
end;

# ╔═╡ 8cd052d9-f40e-4796-aeef-52c02b3bf156
Dx+Dy |> repl_show

# ╔═╡ 1b1b7d12-50ef-4c4e-9376-a353a56540c3
Dx∘Dy |> repl_show

# ╔═╡ 57e48b4c-eef6-433e-98a1-1006e844b368
Dx*v |> repl_show

# ╔═╡ e2bf2649-4770-4f52-9752-b61ce03c6f82
(Dx+Dy)*v |> repl_show

# ╔═╡ d163d363-853d-4d83-a2d3-f8dd6e8f552d
(Dx∘Dy)*v |> repl_show

# ╔═╡ 1c38d3f9-1839-468c-a368-4ef101bd4f18
laplace(g2, stencil_set2) |> repl_show

# ╔═╡ cf84cefb-2dbd-4b8d-880b-47cc350a7c43
laplace(g2, stencil_set4) |> repl_show

# ╔═╡ 67c73667-1f41-47b5-b59a-459787767f29
# laplace(mg, stencil_set2) |> repl_show

# ╔═╡ 4634c1a6-0520-4b0f-8d32-a1fdf2ebaea5
md"""
## Appendix
"""

# ╔═╡ 24788161-b29a-450a-bd35-f9c29e7ded9a
PlutoUI.TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Diffinitive = "5a373a26-915f-4769-bcab-bf03835de17b"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
Diffinitive = "~0.1.5"
PlutoUI = "~0.7.71"
StaticArrays = "~1.9.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "8ad3b11d3caefa740741684109dab3915258eb55"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Diffinitive]]
deps = ["LinearAlgebra", "StaticArrays", "TOML"]
git-tree-sha1 = "064ed50f32f0fb073406c2db1f391faeb1f2357c"
uuid = "5a373a26-915f-4769-bcab-bf03835de17b"
version = "0.1.5"

    [deps.Diffinitive.extensions]
    DiffinitiveMakieExt = "Makie"
    DiffinitivePlotsExt = "Plots"
    DiffinitiveSparseArrayKitExt = ["SparseArrayKit", "Tokens"]
    DiffinitiveSparseArraysExt = ["SparseArrays", "Tokens"]

    [deps.Diffinitive.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    SparseArrayKit = "a9a3c162-d163-4c15-8926-b8794fbefed2"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tokens = "040c2ec2-8d69-4aca-bf03-7d3a7092f2f6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eee1b9ad8b29ef0d936e3ec9838c7ec089620308"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.16"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─885c60d7-d33c-4741-ae49-6a57510ec7b5
# ╠═9ee3372a-e78d-4f74-84ce-e04208d1558d
# ╟─51c02ced-f684-417f-83f1-cade4edda43f
# ╠═25c90528-22cd-41ca-8572-ccd946928318
# ╠═e7f3e466-9833-428c-99ad-20bc9d88d951
# ╠═2e74f9b5-5b4f-4887-8a30-4655d560a45c
# ╠═365524b5-3182-4691-9817-1bbec1492c14
# ╠═d5725e1b-bc4f-4a95-975d-179c193908c9
# ╠═b824ef8d-5026-4861-9a23-45a7939fd38c
# ╠═fbe365a2-f95e-4297-8326-c18d22932869
# ╠═56670aff-0343-41cb-a653-35a61376dda4
# ╠═b5a6491e-a93e-4058-8ceb-be1dc4d4c100
# ╠═828d57a1-ee58-4204-8050-78127821a4c6
# ╠═127d34f6-69f7-4082-a74b-0be86942f153
# ╠═c46a278e-a102-4544-82d8-7df816440410
# ╟─5aa7079c-8005-47f1-bb82-c35f3aa54b42
# ╠═08f493ed-189c-43f3-86f2-95fc475ec0e7
# ╟─f7244bf7-8266-469f-b07f-30c203d9af48
# ╠═0e14bd28-5dd1-44c4-abf4-23b70546bd49
# ╠═fcb74341-6b03-4ada-8f5d-bc245c23679b
# ╠═8dec053b-eaae-463d-800b-b8d89d5d550b
# ╠═c1172a36-c5d7-47dc-bc79-af0d43a8f6ee
# ╟─9c889176-865b-402d-81b5-71957d2878f7
# ╠═85e8e748-e575-4a29-80c7-22d110578343
# ╠═804ad722-9081-4d1d-b0d2-c536a26fe20d
# ╟─a70c689d-0851-497f-938a-e5c92ce59ddb
# ╠═2afde3fe-96ed-4d7e-a79b-fc880e0da268
# ╠═5451a071-14ae-47ae-99c5-4d65508d280f
# ╠═b6b06fe9-de16-41ca-ad45-eef6dd038485
# ╠═d1c8c3a0-76ec-4c32-853e-0471d71e5cf0
# ╠═6051c144-9982-4bd9-92f9-d0aaf3961872
# ╠═12a9f430-f96b-43f2-bf63-149b5a028fd7
# ╠═e44f8d91-1cbf-44be-bef1-40e60c4a777f
# ╠═c378b88a-1d74-44a2-bdc1-b371da478de8
# ╠═9614cda7-b48c-4925-89b5-113cf514f20f
# ╠═725e9430-4821-4939-bfef-6c186d2dc500
# ╠═e8ca54a1-a6db-40a1-b44a-73e175894df4
# ╠═5d1f10fe-f620-469c-822c-55955a5541ad
# ╠═fdc531d4-9d27-41dc-ba79-6febefde223a
# ╠═fff2e04a-357f-4254-996e-d5ccd9ff31f8
# ╠═959b071e-1ef6-4f29-aa8b-d88bfef80c00
# ╠═0a0f7e77-a789-4fb3-a4c2-7853b67788ec
# ╠═177e0893-fbb1-4bb5-a108-e5990e943ab7
# ╠═f1b6bd54-baf6-4360-aec0-bd8d52497894
# ╠═a8f8343e-cd6f-451d-a2b3-e5f0f561f8af
# ╠═50291998-7194-4a0d-9c19-eacc48b3f5da
# ╠═1720af08-85e4-4502-b37e-9fa73008e221
# ╠═4a72f217-2423-4bc1-8452-eb28dde36689
# ╠═48a28ade-73bb-461d-ab96-82f92ed199c8
# ╠═3c681a63-94a6-4677-aef7-df903c463896
# ╠═db735370-153a-40f9-b77f-9f60e30a35c4
# ╠═613ebac7-50bc-424c-8fa2-064b64c93319
# ╠═9e7d7667-960e-491f-8b83-e3b01a0db5b0
# ╠═c3005e74-5b96-4b0c-9c57-b7d02968ed94
# ╠═fd019973-0d54-4e31-b43b-f53a704cb01c
# ╠═df2e8af0-9ca5-4972-8771-f8bea1591f85
# ╠═8e538109-16a1-4af7-a986-9dc1455b7de7
# ╠═5f3744a4-72d8-4448-820f-a928bfaaf825
# ╠═4529b5c4-4905-4fd3-9aaa-5f88faa841c8
# ╠═c3bb0450-a7d5-44c1-9ca9-9e1ecf2db9f8
# ╠═da63e57a-0794-4cd8-9941-ada0e5c1c40e
# ╟─0b951425-979c-4ac9-8581-f690b729bab4
# ╠═a39bb6e2-f1fe-4206-8ee3-88ff0c075233
# ╠═8cd052d9-f40e-4796-aeef-52c02b3bf156
# ╠═1b1b7d12-50ef-4c4e-9376-a353a56540c3
# ╠═57e48b4c-eef6-433e-98a1-1006e844b368
# ╠═e2bf2649-4770-4f52-9752-b61ce03c6f82
# ╠═d163d363-853d-4d83-a2d3-f8dd6e8f552d
# ╠═1c38d3f9-1839-468c-a368-4ef101bd4f18
# ╠═cf84cefb-2dbd-4b8d-880b-47cc350a7c43
# ╠═67c73667-1f41-47b5-b59a-459787767f29
# ╟─4634c1a6-0520-4b0f-8d32-a1fdf2ebaea5
# ╠═1f8a7cfa-94cc-41bf-a8c8-2dc5218741e0
# ╠═24788161-b29a-450a-bd35-f9c29e7ded9a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
