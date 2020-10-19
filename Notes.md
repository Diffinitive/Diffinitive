# Notes

## Known size of range and domain?
Is there any reason to use a trait to differentiate between fixed size and unknown size?

When do we need to know the size of the range and domain?
 * When indexing to provide boundschecking?
 * When doing specialised computations for different parts of the range/domain?
 * More?

 Maybe if we should have dynamic sizing it could be only for the range. `domain_size` would not be implemented. And the `range_size` would be a function of a vector that the TensorMapping is applied to.

## Reasearch and thinking
 - [ ] Use a trait to indicate if a TensorMapping uses indices with regions.
    The default should be that they do NOT.
        - [ ] What to name this trait? Can we call it IndexStyle but not export it to avoid conflicts with Base.IndexStyle?
 - [ ] Use a trait to indicate that a TensorMapping har the same range and domain?
 - [ ] Rename all the Tensor stuff to just LazyOperator, LazyApplication and so on?
 - [ ] Figure out repeated application of regioned TensorMappings. Maybe an instance of a tensor mapping needs to know the exact size of the range and domain for this to work?
 - [ ] Check how the native julia doc generator works
    - [ ] Check if Vidars design docs fit in there
 - [ ] Create a macro @lazy which replaces a binary op (+,-) by its lazy equivalent? Would be a neat way to indicate which evaluations are lazy without cluttering/confusing with special characters.
 - [ ] Specificera operatorer i TOML eller något liknande?
 H.. H_gamma etc.)
 - [ ] Dispatch in Lower() instead of the type Lower so `::Lower` instead of `::Type{Lower}` ???
 	Seems better unless there is some specific reason to use the type instead of the value.
 - [ ] How do we handle mixes of periodic and non-periodic grids? Seems it should be supported on the grid level and on the 1d operator level. Between there it should be transparent.
 - [ ] Can we have a trait to tell if a TensorMapping is transposable?
 - [ ] Is it ok to have "Constructors" for abstract types which create subtypes? For example a Grids() functions that gives different kind of grids based on input?

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

### Alternativ:

#### 1.Använda tuplar
Fördelar:

Nackdelar:

Syntax:
```
f(x,y) = x^2 + y^2
gf = eval_on_grid(g,f)
gf[2,3] # En tupel för en given gridpunkt
gf[2,3][2] # Andra komponenten av vektor fältet i en punkt.

```

#### 2.AbstractArray{T,2} where T
Låta alla saker ta in AbstractArray{T,2} where T. Där T kan vara lite vad som helst, tillexemel en SVector eller Array. Men Differens-opertorerna bryr sig inte om det.

En nackdel kan var hur man ska få ut gridfunktionen för tex andra komponenten.

Syntax:
```
gf = eval(...)
gf[2,3] # Hela vektorn för en gridpunkt
gf[2,3][2] # Andra komponenten av vektor fältet.
```
#### 3.AbstractArray{T,2+1} where T
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
För alternativ 1 och 2 har vi problemet hur vi får ut komponenter som vektorfält. Detta behöver antagligen kunna ske lazy.
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
