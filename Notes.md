# Notes

## Reading operators

Jonatan's suggestion is to add methods to `Laplace`, `SecondDerivative` and
similar functions that take in a filename from which to read stencils. These
methods encode how to use the structure in a file to build the particular
operator. The filename should be a keyword argument and could have a default
value.

 * This allows easy creation of operators without the user having to handle stencils.
 * The user can easily switch between sets of operators by changing the file stecils are read from.

Grids for optimized operators could be created by reading from a .toml file in
a similar fashion as for the operators. The grid can then be used in a
`Laplace` method which dispatches on the grid type and knows how to read the
optimized operators. The method would also make sure the operators match the
grid.

Idea: Make the current upper case methods lower case. Add types with the upper
case names. These types are tensor mappings for the operator but also contain
the associated operators as fields. For example:

```julia
 L = Laplace(grid)
 L.H
 L.Hi
 L.e
 L.d
 L.M

 wave = L - L.Hi∘L.e'∘L.d
```

These types could also contain things like borrowing and such.

## Storage of operators
We need to change the toml format so that it is easier to store several
operator with different kinds of differentiations. For example there could be
several operators of the same order but with different number of boundary
points or different choice of boundary stencils.

Properties that differentiate operators should for this reason be stored in
variables and not be section or table names.

Operators/sets of stencils should be stored in an [array of tables](https://toml.io/en/v1.0.0-rc.3#array-of-tables).

We should formalize the format and write simple and general access methods for
getting operators/sets of stencils from the file. They should support a simple
way to filter based on values of variables. There filters could possibly be
implemented through keyword arguments that are sent through all the layers of
operator creation.

* Remove order as a table name and put it as a variable.

### Parsing of stencil sets
At the moment the only parsing that can be done at the top level is conversion
from the toml file to a dict of strings. This forces the user to dig through
the dictionary and apply the correct parsing methods for the different parts,
e.g. `parse_stencil` or `parse_tuple`. While very flexible there is a tight
coupling between what is written in the file and what code is run to make data
in the file usable. While this coupling is hard to avoid it should be made
explicit. This could be done by putting a reference to a parsing function in
the operator-storage format or somehow specifying the type of each object.
This mechanism should be extensible without changing the package. Perhaps
there could be a way to register parsing functions or object types for the
toml.

If possible the goal should be for the parsing to get all the way to the
stencils so that a user calls `read_stencil_set` and gets a
dictionary-structure containing stencils, tuples, scalars and other types
ready for input to the methods creating the operators.

## Known size of range and domain?
Is there any reason to use a trait to differentiate between fixed size and unknown size?

When do we need to know the size of the range and domain?
 * When indexing to provide boundschecking?
 * When doing specialised computations for different parts of the range/domain?
 * More?

 Maybe if we should have dynamic sizing it could be only for the range. `domain_size` would not be implemented. And the `range_size` would be a function of a vector that the LazyTensor is applied to.

## Reasearch and thinking
 - [ ] Use a trait to indicate that a LazyTensor har the same range and domain?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Create a macro @lazy which replaces a binary op (+,-) by its lazy equivalent? Would be a neat way to indicate which evaluations are lazy without cluttering/confusing with special characters.
 - [ ] Dispatch on Lower() instead of the type Lower so `::Lower` instead of `::Type{Lower}` ???
 	Seems better unless there is some specific reason to use the type instead of the value.
 - [ ] Can we have a trait to tell if a LazyTensor is transposable?
 - [ ] Is it ok to have "Constructors" for abstract types which create subtypes? For example a Grids() functions that gives different kind of grids based on input?
 - [ ] Figure out how to treat the borrowing parameters of operators. Include in into the struct? Expose via function dispatched on the operator type and grid?

## Identifiers for regions
The identifiers (`Upper`, `Lower`, `Interior`) used for region indecies should probably be included in the grid module. This allows new grid types to come with their own regions.
We implement this by refactoring RegionIndices to be agnostic to the region types and then moving the actual types to Grids.

## Regions and tensormappings
- [ ] Use a trait to indicate if a LazyTensor uses indices with regions.
    The default should be that they do NOT.
        - [ ] What to name this trait? Can we call it IndexStyle but not export it to avoid conflicts with Base.IndexStyle?
 - [ ] Figure out repeated application of regioned LazyTensors. Maybe an instance of a tensor mapping needs to know the exact size of the range and domain for this to work?

### Ideas for information sharing functions
```julia
using StaticArrays

function regions(op::SecondDerivativeVariable)
    t = ntuple(i->(Interior(),),range_dim(op))
    return Base.setindex(t, (Lower(), Interior(), Upper()), derivative_direction(op))
end

function regionsizes(op::SecondDerivativeVariable)
    sz = tuple.(range_size(op))

    cl = closuresize(op)
    return Base.setindex(sz, (cl, n-2cl, cl), derivative_direction(op))
end


g = EquidistantGrid((11,9), (0.,0.), (10.,8.)) # h = 1
c = evalOn(g, (x,y)->x+y)

D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,1)
@test regions(D₂ᶜ) == (
    (Lower(), Interior(), Upper()),
    (Interior(),),
)
@test regionsizes(D₂ᶜ) == ((1,9,1),(9,))


D₂ᶜ = SecondDerivativeVariable(g, c, interior_stencil, closure_stencils,2)
@test regions(D₂ᶜ) == (
    (Interior(),),
    (Lower(), Interior(), Upper()),
)
@test regionsizes(D₂ᶜ) == ((11,),(1,7,1))
```


## Boundschecking and dimension checking
Does it make sense to have boundschecking only in getindex methods?
This would mean no bounds checking in applys, however any indexing that they do would be boundschecked. The only loss would be readability of errors. But users aren't really supposed to call apply directly anyway.

Preferably dimensions and sizes should be checked when lazy objects are created, for example TensorApplication, TensorComposition and so on. If dimension checks decreases performance we can make them skippable later.

## Changes to `eval_on`
There are reasons to replace `eval_on` with regular `map` from Base, and implement a kind of lazy map perhaps `lmap` that work on indexable collections.

The benefit of doing this is that we can treat grids as gridfunctions for the coordinate function, and get a more flexible tool. For example `map`/`lmap` can then be used both to evaluate a function on the grid but also get a component of a vector valued grid function or similar.

A question is how and if we should implement `map`/`lmap` for functions like `(x,y)->x*y` or stick to just using vector inputs. There are a few options.

* use `Base.splat((x,y)->x*y)` with the single argument `map`/`lmap`.
* implement a kind of `unzip` function to get iterators for each component, which can then be used with the multiple-iterators-version of `map`/`lmap`.
* Inspect the function in the `map`/`lmap` function to determine which matches.

Below is a partial implementation of `lmap` with some ideas
```julia
struct LazyMapping{T,IT,F}
    f::F
    indexable_iterator::IT # ___
end

function LazyMapping(f,I)
    IT = eltype(I)
    T = f(zero(T))
    F = typeof(f)

    return LazyMapping{T,IT,F}(f,I)
end

getindex(lm::LazyMapping, I...) = lm.f(lm.I[I...])
# indexabl interface
# iterable has shape

iterate(lm::LazyMapping) = _lazy_mapping_iterate(lm, iterate(lm.I))
iterate(lm::LazyMapping, state) = _lazy_mapping_iterate(lm, iterate(lm.I, state))

_lazy_mapping_iterate(lm, ::Nothing) = nothing
_lazy_mapping_iterate(lm, (next, state)) = lm.f(next), state

lmap(f,  I) = LazyIndexableMap(f,I)
```

The interaction of the map methods with the probable design of multiblock functions involving nested indecies complicate the picture slightly. It's clear at the time of writing how this would work with `Base.map`. Perhaps we want to implement our own versions of both eager and lazy map.


### 2024-04
MappedArrays.jl provides a simple array type and function like the description
of LazyMapping above. One option is to remove `eval_on` completely and rely on
destructuring arguments if handling the function input as a vector is
undesirable.

If we can let multi-block grids be iterators over grid points we could even
handle those by specialized implementation of `map` and `mappedarray`.

## Multiblock implementation
We want multiblock things to work very similarly to regular one block things.

### Grid functions
Should probably support a nested indexing so that we first have an index for
subgrid and then an index for nodes on that grid. E.g `g[1,2][2,3]` or
`g[3][43,21]`.

We could also possibly provide a combined indexing style `g[1,2,3,4]` where
the first group of indices are for the subgrid and the remaining are for the
nodes.

We should make sure the underlying buffer for grid functions are continuously
stored and are easy to convert to, so that interaction with for example
DifferentialEquations is simple and without much boilerplate.

#### `map` and `collect` and nested indexing
We need to make sure `collect`, `map` and a potential lazy map work correctly
through the nested indexing. Also see notes on `eval_on` above.

Possibly this can be achieved by providing special nested indexing but not
adhering to an array interface at the top level, instead being implemented as
an iterator over the grid points. A custom trait can let map and other methods
know the shape (or structure) of the nesting so that they can efficiently
allocate result arrays.

### Tensor applications
Should behave as grid functions

### LazyTensors
Could be built as a tuple or array of LazyTensors for each grid with a simple apply function.

Nested indexing for these is problably not needed unless it simplifies their own implementation.

Possibly useful to provide a simple type that doesn't know about connections between the grids. Antother type can include knowledge of the.

We have at least two option for how to implement them:
* Matrix of LazyTensors
* Looking at the grid and determining what the apply should do.

### Overall design implications of nested indices
If some grids accept nested indexing there might be a clash with how LazyArrays work. It would be nice if the grid functions and lazy arrays that actually are arrays can be AbstractArray and things can be relaxed for nested index types.

## Vector valued grid functions

### Test-applikationer
div- och grad-operationer

Enligt Wikipedia verkar det som att `∇⋅` agerar på första dimensionen av ett tensorfält och `div()` på sista.
Om man generaliserar kanske `∇` i så fall bara lägger till en dimension i början.

Kan vi implementera `⋅`(\cdot) så att de fungerar som man vill för både tensor-fält och tensor-operatorer?

Är `∇` ett tensor-fält av tensor-operatorer? Vad är ett tensor-fält i vår kod? Är det en special-fall av en tensor-mapping?

### Grid-funktionen
Grid-funktioner har typen `AbstractArray{T,2} where T`.
`T` kan vara lite vad som helst, tillexemel en SVector eller Array, eller Tuple. Tensoroperatorerna bryr sig inte om exakt vad det är, mer än att typen måste stödja de operationer som operatorn använder.

En nackdel kan vara hur man ska få ut gridfunktionen för tex andra komponenten.

Syntax:
```
f(x̄) = x̄
gf = evalOn(g, f)
gf[2,3] # x̄ för en viss gridpunkt
gf[2,3][2] # x̄[2] för en viss gridpunkt
```

### Tensor operatorer
Vi kan ha tensor-operatorer som agerar på ett skalärt fält och ger ett vektorfält eller tensorfält.
Vi kan också ha tensor-operatorer som agerar på ett vektorfält eller tensorfält och ger ett skalärt fält.

TBD: Just nu gör `apply_transpose` antagandet att domän-typen är samma som range-typen. Det behöver vi på något sätt bryta. Ett alternativ är låta en LazyTensor ha `T_domain` och `T_range` istället för bara `T`. Känns dock lite grötigt. Ett annat alternativ skulle vara någon typ av trait för transpose? Den skulle kunna innehålla typen som transponatet agerar på? Vet inte om det fungerar dock.

TBD: Vad är målet med `T`-parametern för en LazyTensor? Om vi vill kunna applicera en difference operator på vad som helst kan man inte anta att en `LazyTensor{T}` bara agerar på instanser av `T`.

Man kan implementera `∇` som en tensormapping som agerar på T och returnerar `StaticVector{N,T} where N`.
(Man skulle eventuellt också kunna låta den agera på `StaticMatrix{N,T,D} where N` och returnera `StaticMatrix{M,T,D+1}`. Frågan är om man vinner något på det...)

Skulle kunna ha en funktion `range_type(::LazyTensor, ::Type{domain_type})`

Kanske kan man implementera `⋅(tm::LazyTensor{R,D}, v::AbstractArray{T,D})` där T är en AbstractArray, tm på något sätt har komponenter, lika många som T har element.

### Komponenter som gridfunktioner
En viktig operation för vektorfält är att kunna få ut komponenter som grid-funktioner. Detta behöver antagligen kunna ske lazy.
Det finns ett par olika lösningar:
* Använda map eller en lazy map (se diskussion om eval_on)
* Implementera en egen typ av view som tar hand om detta. Eller Accessors.jl?
* Använda en LazyTensor
* Någon typ av lazy-broadcast
* En lazy array som applicerar en funktion för varje element.


### Prestanda-aspekter
[Vidar, Discord, 2023-03-03]
Typiskt sett finns det två sätt att representera vektorvärda gridfunktioner AbstractArray{T,Dim} där T är en vektor över komponenterna. Man skulle alltså i 1D ha
u = [ [u1[x1], u2[x1]] , [u1[x2], u2[x2]], ... [u1[xN], u2[xN]]]. Detta brukar kallas array of structs (AoS). Alternativet är struct of arrays (SoA), där man har alla gridpunkter för en given komponent u = [[u1[x1], u1[x2]],... u1[xN]], [u2[x1], u2[x2], ... u2[xN]]].

Personligen tycker jag att AoS känns som den mer naturliga representationen? Det skulle göra det enklarare att parallelisera en vektorvärd gridfunktion över gridpunkterna, och om man opererar på olika komponenter i samma funktion så är det också bra ur en minnesaccess-synpunkt då dessa kommer ligga nära vandra i minnet. Problemet är att AoS sabbar vektorisering på CPU då två gridpunkter i en komponent ligger långt bort från varandra. Efter lite eftersökningar (och efter att snackat lite med Ossian) så verkar det ändå som att AoS är dåligt på GPU, där man vill att trådar typiskt sett utföra samma operation på närliggande minne.

Vad tänker du kring detta ur ett interface-perspektiv? Jag hittade paketet  https://github.com/JuliaArrays/StructArrays.jl som verkar erbjuda AoS-interface men SoA-minneslayout så det kanske kan vara något vi kan använda? Inte native-stödd på samma sätt som SVector, men verkar iaf utvecklas aktivt.

[Efter telefonsamtal] För optimal prestanda behöver vi antagligen se till att man kan räkna ut varje komponent i en punkt individuellt. Detta så att man har frihet att till exempel låta den innersta loopen hålla komponentindexet konstant för att underlätta intruktionsvektorisering.


[Vidare tankar]
 * Det borde bara vara output-gridfunktionen som behöver special-indexeras? Det viktiga på inputsidan är att den är lagrad på rätt sätt i minnet.
 * Det borde inte vara några problem att behålla det "optimala" interfacet (gf[1,1,1][2]) till gridfunktionerna. Om man verkligen behöver kan skapa parallella indexeringsmetoder som gör det man behöver, i.e, "deep indexing".
 * Det är inte säkert att vi behöver göra något speciellt på outputsidan överhuvudtaget. Det känns inte orimligt att kompilatorn skulle kunna optimera bort den koden som räknar ut onödiga komponenter.
 * Om vi behöver special-indexering kommer till exempel LazyTensorApplication att behöva implementera det.
 * För att komma vidare med något mer avancerat behöver vi antagligen implementera några operatorer som ger och agerar på vektorvärda funktioner. Tex grad, elastiska operatorn, andra?


## Performance measuring
We should be measuring performance early. How does our effective cpu and memory bandwidth utilization compare to peak performance?

We should make these test simple to run for any solver.

See [this talk](https://www.youtube.com/watch?v=vPsfZUqI4_0) for some simple ideas for defining effecive memory usage and some comparison with peak performance.


## Adjoint as a trait on the sbp_operator level?

It would be nice to have a way of refering to adjoints with resepct to the sbp-inner-product.
If it was possible you could reduce the number of times you have to deal with the inner product matrix.

Since the LazyOperators package is sort of implementing matrix-free matrices there is no concept of inner products there at the moment. It seems to complicate large parts of the package if this was included there.

A different approach would be to include it as a trait for operators so that you can specify what the adjoint for that operator is.


## Name of the `VolumeOperator` type for constant stencils
It seems that the name is too general. The name of the method `volume_operator` makes sense. It should return different types of `LazyTensor` specialized for the grid. A suggetion for a better name is `ConstantStencilVolumeOperator`


## Implementation of LazyOuterProduct
Could the implementation of LazyOuterProduct be simplified by making it a
struct containing two or more LazyTensors? (using split_tuple in a similar way
as TensorGrid)

## Implementation of boundary_indices for more complex grids
To represent boundaries of for example tet-elements we can use a type `IndexCollection` to index a grid function directly.

```julia
I = IndexCollection(...)
v[I]
```

* This would impact how tensor grid works.
* To make things homogenous maybe these index collections should be used for the more simple grids too.
* The function `to_indices` from Base could be useful to implement for `IndexCollection`


## Stencil application pipeline
We should make sure that `@inbounds` and `Base.@propagate_inbounds` are
applied correctly throughout the stack. When testing the performance of
stencil application on the bugfix/sbp_operators/stencil_return_type branch
there seemed to be some strange results where such errors could be the
culprit.
