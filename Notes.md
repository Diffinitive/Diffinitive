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


## Variable second derivative

2020-12-08 after discussion with Vidar:
We will have to handle the variable second derivative in a new variant of
VolumeOperator, "SecondDerivativeVariable?". Somehow it needs to know about
the coefficients. They should be provided as an AbstractVector. Where they are
provided is another question. It could be that you provide a reference to the
array to the constructor of SecondDerivativeVariable. If that array is mutable
you are free to change it whenever and the changes should propagate
accordingly. Another option is that the counter part to "Laplace" for this
variable second derivate returns a function or acts like a functions that
takes an Abstract array and returns a SecondDerivativeVariable with the
appropriate array. This would allow syntax like `D2(a)*v`. Can this be made
performant?

For the 1d case we can have a constructor
`SecondDerivativeVariable(D2::SecondDerivativeVariable, a)` that just creates
a copy with a different `a`.

Apart from just the second derivative in 1D we need operators for higher
dimensions. What happens if a=a(x,y)? Maybe this can be solved orthogonally to
the `D2(a)*v` issue, meaning that if a constant nD version of
SecondDerivativeVariable is available then maybe it can be wrapped to support
function like syntax. We might have to implement `SecondDerivativeVariable`
for N dimensions which takes a N dimensional a. If this could be easily
closured to allow D(a) syntax we would have come a long way.

For `Laplace` which might use a variable D2 if it is on a curvilinear grid we
might want to choose how to calculate the metric coefficients. They could be
known on closed form, they could be calculated from the grid coordinates or
they could be provided as a vector. Which way you want to do it might change
depending on for example if you are memory bound or compute bound. This choice
cannot be done on the grid since the grid shouldn't care about the computer
architecture. The most sensible option seems to be to have an argument to the
`Laplace` function which controls how the coefficients are gotten from the
grid. The argument could for example be a function which is to be applied to
the grid.

What happens if the grid or the varible coefficient is dependent on time?
Maybe it becomes important to support `D(a)` or even `D(t,a)` syntax in a more
general way.

```
g = TimeDependentGrid()
L = Laplace(g)
function Laplace(g::TimeDependentGrid)
    g_logical = logical(g) # g_logical is time independent
    ... Build a L(a) assuming we can do that ...
    a(t) = metric_coeffs(g,t)
    return t->L(a(t))
end
```

## Known size of range and domain?
Is there any reason to use a trait to differentiate between fixed size and unknown size?

When do we need to know the size of the range and domain?
 * When indexing to provide boundschecking?
 * When doing specialised computations for different parts of the range/domain?
 * More?

 Maybe if we should have dynamic sizing it could be only for the range. `domain_size` would not be implemented. And the `range_size` would be a function of a vector that the TensorMapping is applied to.

## Reasearch and thinking
 - [ ] Use a trait to indicate that a TensorMapping har the same range and domain?
 - [ ] Rename all the Tensor stuff to just LazyOperator, LazyApplication and so on?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Create a macro @lazy which replaces a binary op (+,-) by its lazy equivalent? Would be a neat way to indicate which evaluations are lazy without cluttering/confusing with special characters.
 - [ ] Specificera operatorer i TOML eller något liknande?
 H.. H_gamma etc.)
 - [ ] Dispatch on Lower() instead of the type Lower so `::Lower` instead of `::Type{Lower}` ???
 	Seems better unless there is some specific reason to use the type instead of the value.
 - [ ] How do we handle mixes of periodic and non-periodic grids? Seems it should be supported on the grid level and on the 1d operator level. Between there it should be transparent.
 - [ ] Can we have a trait to tell if a TensorMapping is transposable?
 - [ ] Is it ok to have "Constructors" for abstract types which create subtypes? For example a Grids() functions that gives different kind of grids based on input?

## Regions and tensormappings
- [ ] Use a trait to indicate if a TensorMapping uses indices with regions.
    The default should be that they do NOT.
        - [ ] What to name this trait? Can we call it IndexStyle but not export it to avoid conflicts with Base.IndexStyle?
 - [ ] Figure out repeated application of regioned TensorMappings. Maybe an instance of a tensor mapping needs to know the exact size of the range and domain for this to work?

## Boundschecking and dimension checking
Does it make sense to have boundschecking only in getindex methods?
This would mean no bounds checking in applys, however any indexing that they do would be boundschecked. The only loss would be readability of errors. But users aren't really supposed to call apply directly anyway.

Preferably dimensions and sizes should be checked when lazy objects are created, for example TensorApplication, TensorComposition and so on. If dimension checks decreases performance we can make them skippable later.

## Vector valued grid functions
Från slack konversation:

Jonatan Werpers:
Med vektorvärda gridfunktioner vill vi ju fortfarande att grid funktionen ska vara till exempel AbstractArray{LitenVektor,2}
Och att man ska kunna göra allt man vill med LitenVektor
typ addera, jämföra osv
Och då borde points returnera AbstractArray{LitenVektor{Float,2},2} för ett 2d nät
Men det kanske bara ska vara Static arrays?

Vidar Stiernström:
Ja, jag vet inte riktigt vad som är en rimlig representation
Du menar en vektor av static arrays då?

Jonatan Werpers:
Ja, att LitenVektor är en StaticArray

Vidar Stiernström:
Tuplar känns typ rätt inuitivt för att representera värdet i en punkt
men
det suger att man inte har + och - för dem

Jonatan Werpers:
Ja precis

Vidar Stiernström:
så kanske är bra med static arrays i detta fall

Jonatan Werpers:
Man vill ju kunna köra en Operator rakt på och vara klar eller?

Vidar Stiernström:
Har inte alls tänkt på hur det vi gör funkar mot vektorvärda funktioner
men känns som staticarrays är hur man vill göra det
tuplar är ju immutable också
blir jobbigt om man bara agerar på en komponent då

Jonatan Werpers:
Hm…
Tål att tänkas på
Men det lär ju bli mer indirektion med mutables eller?
Hur fungerar det?
Det finns ju hur som helst både SVector och MVector i StaticArrays

Vidar Stiernström:
När vi jobbat i c/c++ och kollat runt lite hur man brukar göra så lagrar man i princip alla sina obekanta i en lång vektor och så får man specificera i funktioerna vilken komponent man agerar på och till vilken man skriver
så man lagrar grejer enl: w = [u1, v1, u2, v2, …] i 1D.
Men alltså har ingen aning hur julia hanterar detta

Jonatan Werpers:
Det vi är ute efter kanske är att en grid funcktion är en AbstractArray{T,2} where T<:NågotSomViKanRäknaMed
Och så får den typen var lite vad som helst.

Vidar Stiernström:
Tror det kan vara farligt att ha nåt som är AbstractArray{LitenArray{NDof},Dim}
Jag gissar att det kompilatorn vill ha är en stor array med doubles

Jonatan Werpers:
Och sen är det upp till den som använder grejerna att vara smart
Vill man vara trixig kan man väl då imlementera SuperHaxxorGridFunction <: AbstractArray{Array{…},2} som lagrar allt linjärt eller något sånt
Det kommer väl lösa sig när man börjar implementera vektorvärda saker
Euler nästa!
New
Vidar Stiernström:
Det vore skönt att inte behöva skriva såhär varje gång man testar mot en tupel :smile: @test [gp[i]...] ≈ [p[i]...] atol=5e-13

Jonatan Werpers:
https://github.com/JuliaArrays/ArraysOfArrays.jl
https://github.com/jw3126/Setfield.jl

### Test-applikationer
div och grad operationer

Enligt Wikipedia verkar det som att `∇⋅` agerar på första dimensionen av ett tensor fält och `div()` på sista.
Om man generaliserar kanske `∇` i så fall bara lägger till en dimension i början.

Kan vi implementera `⋅`(\cdot) så att de fungerar som man vill för både tensor-fält och tensor-operatorer?

Är `∇` ett tensor-fält av tensor-operatorer? Vad är ett tensor-fält i vår kod? Är det en special-fall av en tensor-mapping?

### Grid-funktionen
Grid-funktionon har typen `AbstractArray{T,2} where T`.
`T` kan vara lite vad som helst, tillexemel en SVector eller Array, eller tuple. TensorOperatorerna bryr sig inte om exakt vad det är, mer än att typen måste stödja de operationer som operatorn använder.

En nackdel kan vara hur man ska få ut gridfunktionen för tex andra komponenten.

Syntax:
```
f(x̄) = x̄
gf = evalOn(g, f)
gf[2,3] # x̄ för en viss gridpunkt
gf[2,3][2] # x̄[2] för en viss gridpunkt
```

Note: Behöver bestämma om eval on skickar in `x̄` eller `x̄...` till `f`. Eller om man kan stödja båda.

### Tensor operatorer
Vi kan ha tensor-operatorer som agerar på ett skalärt fält och ger ett vektorfält eller tensorfält.
Vi kan också ha tensor-operatorer som agerar på ett vektorfält eller tensorfält och ger ett skalärt fält.

TBD: Just nu gör `apply_transpose` antagandet att domän-typen är samma som range-typen. Det behöver vi på något sätt bryta. Ett alternativ är låta en TensorMapping ha `T_domain` och `T_range` istället för bara `T`. Känns dock lite grötigt. Ett annat alternativ skulle vara någon typ av trait för transpose? Den skulle kunna innehålla typen som transponatet agerar på? Vet inte om det fungerar dock.

TBD: Vad är målet med `T`-parametern för en TensorMapping? Om vi vill kunna applicera en difference operator på vad som helst kan man inte anta att en `TensorMapping{T}` bara agerar på instanser av `T`.

Man kan implementera `∇` som en tensormapping som agerar på T och returnerar `StaticVector{N,T} where N`.
(Man skulle eventuellt också kunna låta den agera på `StaticMatrix{N,T,D} where N` och returnera `StaticMatrix{M,T,D+1}`. Frågan är om man vinner något på det...)

Skulle kunna ha en funktion `range_type(::TensorMapping, ::Type{domain_type})`

Kanske kan man implementera `⋅(tm::TensorMapping{R,D}, v::AbstractArray{T,D})` där T är en AbstractArray, tm på något sätt har komponenter, lika många som T har element.

### Ratade alternativ:


#### 2.AbstractArray{T,2+1} where T (NOPE!)
Blir inte den här. Bryter mot alla tankar om hur grid funktioner ska fungera. Om de tillåts ha en annan dimension än nätet blir allt hemskt.

Man låter helt enkelt arrayen ha en extra dimension. En fördel är att man har en väldigt "native" typ. En nackdel kan vara att det eventuellt blir rörigt vilken dimension olika operatorer ska agera på. I värsta fall behöver vi "kroneckra in" de tillagda dimensionerna. Vektorfältets index kommer också att bli det första eftersom vi vill att de ska lagras kontinuerligt i minnet pga chachen. (Går kanske att lösa med en custom typ men då krånglar man till det för sig). En fördel skulle vara att man enkelt får ut olika komponenter.

Syntax:
```
gf = eval_on_grid(g,f)
gf[:,2,3] # Hela vektorn för en gridpunkt
gf[2,2,3] # Andra komponenten av vektor fältet i en punkt.
gf[2,:,:] #
```

### Evaluering av funktioner på nät
Hur ska man skriva funktioner som evalueras på nätet? `f(x,y) = ...` eller `f(x̄) = ...`? Eller båda? Kan eval_on_grid se skillnad eller får användaren specificera?

```
f(x,y) = [x^2, y^2]
f(x̄) = [x̄[1]^2, x̄[2]^2]
```

Påverkas detta av hur vi förväntar oss kunna skapa lata gridfunktioner?

### Komponenter som gridfunktioner
En viktig operation för vektor fält är att kunna få ut komponenter som grid-funktioner. Detta behöver antagligen kunna ske lazy.
Det finns ett par olika lösningar:
* Implementera en egen typ av view som tar hand om detta. Eller Accessors.jl?
* Använda en TensorMapping
* Någon typ av lazy-broadcast
* En lazy array som applicerar en funktion för varje element.

Skulle vara en fördel om det är hyffsat generiskt så att en eventuell användare kan utöka det enkelt om de har någon egen exotisk typ. Eller ska man vila helt på

Syntax:
```
gf = eval(...)
component(gf,2) # Andra komponenten av en vektor
component(gf,2,3) # (2,3) elementet av en matris
component(gf,:,2) # Andra kolumnen av en matris
@ourview gf[:,:][2]
```

## Grids embedded in higher dimensions

For grids generated by asking for boundary grids for a regular grid, it would
make sense if these grids knew they were embedded in a higher dimension. They
would return coordinates in the full room. This would make sense when
drawing points for example, or when evaluating functions on the boundary.

Implementation of this is an issue that requires some thought. Adding an extra
"Embedded" type for each grid would make it easy to understand each type but
contribute to "type bloat". On the other hand adapting existing types to
handle embeddedness would complicate the now very simple grid types. Are there
other ways of doing the implentation?

## Performance measuring
We should be measuring performance early. How does our effective cpu and memory bandwidth utilization compare to peak performance?

We should make these test simple to run for any solver.

See [this talk](https://www.youtube.com/watch?v=vPsfZUqI4_0) for some simple ideas for defining effecive memory usage and some comparison with peak performance.


