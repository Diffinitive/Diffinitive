# Grids refactor

## Goals before merging
* A somewhat clear path towards multi-block grids and their grid functions.
* A somewhat clear path towards implementations of div() and rot() or the elastic operator (See Notes.md)

## Change summary
* `EquidistantGrid` is now only a 1D thing.
* Higher dimensions are supported through `TensorGrid`.
* The old behavior of `EquidistantGrid` has been moved to the function `equidistant_grid`.
* Grids embedded in higher dimensions are now supported through tensor products with `ZeroDimGrid`s.
* Vector valued grid functions are now supported and the default element type is `SVector`.
* Grids are now expected to support Julia's indexing and iteration interface.


## TODO
* Document the expected behavior of grid functions
* Write down the thinking around Grid being an AbstractArray. Why it doesn't work

* Clean out Notes.md of any solved issues
* Delete this document, move remaining notes to Notes.md

## Remaining work for feature branches
* Multi-block grids
* Periodic grids
* Grids with modified boundary nodes
* Unstructured grids?

## Frågor

### Should we move utility functions to their own file?

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

### Should Grid have function for the target manifold dimension?
Where would it be used?
    In the constructor for TensorGrid
    In eval on if we want to allow multiargument functions
    Elsewhere?

An alternative is to analyze T in Grid{T,D} to find the answer. (See combined_coordinate_vector_type in tensor_grid.jl)

### Lazy version of map for our needs?
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

### Do we need functions like `getcomponent`?
Perhaps this can be more cleanly solved using map or a lazy version of map?
That approach would be more flexible and more general requiring few specialized functions.

(see "Lazy version of map for our needs?" above)

### Would it help to introduce a type for grid functions?
Seems easier to avoid this but it might be worth investigating.

Can it be done with some kind of trait? We can give AbstractArray the appropriate trait and keep them for the simplest grid functions.

