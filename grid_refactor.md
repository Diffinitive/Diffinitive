# Grids refactor

## Mål
  * Embedded grids
  * Vektorvärda grid funktioner, speciellt koordinat-funktionen
  * Olika griddar i olika riktningar?
      * Tex periodiskt i en riktning och intervall i en.
      * Tex ostrukturerat i ett par och strukturerat i en.
  * Enkelt att evaluera på 0d-grid

## Frågor


### Ska `Grid` vara en AbstractArray?
Efter som alla nät ska agera som en gridfunktion för koordinaterna måste man
svara på frågan hur vi hanterar generellla gridfunktioner samtidigt.

Några saker att förhålla sig till:
  - Multiblock nät?
  - Unstructured?
  - Triangular structured grids?
  - Non-simply connected?
  - CG/DG-nät


Om alla nät är någon slags AbstractArray så kan tillexempel ett multiblock nät vara en AbstractArray{<:Grid, 1} och motsvarande gridfunktioner AbstractArray{<:AbstractArray,1}.
Där man alltså först indexerar för vilket när man vill åt och sedan indexerar nätet. Tex `mg[2][32,12]`.

Ett ostrukturerat nät med till exempel trianglar skulle vi kunna se på samma sätt som ett multiblocknät. Antagligen har de två typerna av nät olika underliggande datastruktur anpassade efter ändamålet.

Hur fungerar tankarna ovan om man har tex tensorprodukten av ett ostrukturerat nät och en ekvidistant nät?
```julia
m = Mesh2DTriangle(10)
e = EqudistantGrid(range(1:10)

e[4] # fourth point

m[3][5] # Fifth node in third triangle
m[3,5] # Fifth node in third triangle # Funkar bara om alla nät är samma, (stämmer inte i mb-fallet)

g = TensorGrid(m, e)

g[3,4][5] # ??
g[3,4] # ??

g[3,5,4] # ??



```

Alla grids kanske inte är AbstractArray? Måste de vara rektangulära? Det blir svårt för strukturerade trianglar och NSC-griddar. Men de ska i allafall vara indexerbara?

Man skulle kunna utesluta MultiblockGrid i tensorgrids

CG-nät och DG-nät blir olika.
På CG-nät kanske man både vill indexera noder och trianglar beroende på vad man håller på med?


Om griddarna inte ska vara AbstractArray finns det många andra ställen som blir konstiga om de är AbstractArray. TensorApplication?! LazyArrays?! Är alla saker vi jobbar med egentligen mer generella object? Finns det något sätt att uttrycka koden så att man kan välja?


Det vi är ute efter är kanske att griddarna uppfyller Iteration och Indexing interfacen.

#### Försök till slutsater
 * Multiblock-nät indexeras i två nivåer tex `g[3][3,4]`
     * Vi struntar i att implementera multiblock-nät som en del av ett tensorgrid till att börja med.
 * En grid kan inte alltid vara en AbstractArray eftersom till exempel ett NCS eller strukturerad triangel inte har rätt form.
 * Om vi har nod-indexerade ostrukturerade nät borde de fungera med TensorGrid.
 * Griddar ska uppfylla Indexing och Iteration interfacen

### Kan vi introducera 1d griddar och tensorgriddar?
  * Vanligt intervallgrid
  * Periodiskt grid
  * 0-dimensionellt grid

Det skulle kunna lösa frågan med embedded-griddar
och frågan om mixade grids.

En svårighet kan vara att implemetera indexeringen för tensorgridden. Är det
lätt att göra den transparent för kompilatorn?

Läs i notes.md om: Grids embedded in higher dimensions

Periodiska gridfunktioner? Eller ska vi bara skita i det och låta användaren
hantera det själv? Blir det krångligt i högre dimensioner?


### Hur ska vi hantera gridfunktioner med flera komponenter?
Tex koordinat-funktionen på nätet?

Funkar det att ha StaticArray som element type?
    Kan man köra rakt på med en operator då?

Vi skulle också kunna använda ArraysOfArrays. Skillnaden blir att den elementtypen är Array{T,N}. Vilket betyder att den är by reference?
    Frågan är om kompilatorn kommer att bry sig... Jag skulle luta mot StaticArrays for safety

Läs i notes.md om: Vector valued grid functions

## Should Grid have function for the target manifold dimension?
Where would it be used?
    In the constructor for TensorGrid
    In eval on if we want to allow multiargument functions
    Elsewhere?

An alternative is to analyze T in Grid{T,D} to find the answer. (See combined_coordinate_vector_type in tensor_grid.jl)

## Lazy version of map for our needs?
Could be used to
 * evaulate functions on grids
 * pick out components of grid functions
 * More?

Maybe this:
```julia
struct LazyMappedArray <: LazyArray
    f::F
    v::AT
end
```

Could allow us to remove eval_on.

## Do we need functions like `getcomponent`?
Perhaps this can be more cleanly solved using map or a lazy version of map?
That approach would be more flexible and more general requiring few specialized functions.

(see "Lazy version of map for our needs?" above)

## Notes from pluto notebook
- Är det dåligt att använda ndims om antalet index inte matchar?
   - Tex ostrukturerat grid
   - Baserat på hur Array fungerar och ndims docs borde den nog returnera
     antalet index och inte griddens dimension
   - Å andra sidan kanske man inte behöver veta viken dimension det är? Fast det känns konstigt...
- Should we keep the `points` function?
   - Maybe it's better to support `collect` on the grid?
- How to handle boundary identifiers and boundary grids on `TensorGrid`?


