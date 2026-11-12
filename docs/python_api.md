# Python API

The Python layer mirrors the CLI and adds programmatic access to parsed
VASP data with persistent caching.

## VASP directory handling: `MRFVaspDir`

{py:class}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir` gives
dictionary-like access to a VASP directory with a persistent disk cache,
so repeated analysis of large trees does not re-parse XML or OUTCAR files
every time.

The cache is optimized for supercomputer file system, avoiding
excessive IO.

```python
from matrifflow.pymatgen.io.vasp.vaspdir import MRFVaspDir

vdir = MRFVaspDir("path/to/vasp/calculation")

structure = vdir.final_structure
energy = vdir.final_energy
incar = vdir["INCAR"]

if vdir.converged:
    print("Calculation converged")

# NEB directories
if vdir.nebp:
    for image_dir in vdir.neb_dirs():
        print(f"Image {image_dir.path}: {image_dir.final_energy}")
```

The example reads three properties:
{py:attr}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.final_structure`,
{py:attr}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.final_energy`,
and {py:attr}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.converged`.
NEB image directories come from
{py:attr}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.nebp` and
{py:meth}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.neb_dirs`.

{py:attr}`~matrifflow.pymatgen.io.vasp.vaspdir.MRFVaspDir.final_energy_reliable`
returns a float only when the energy is judged usable, otherwise
`"unreliable"` (for example, a volume/shape relaxation) or
`"unconverged"`.

The cache is change-aware: it stores a file signature and re-parses only
directories whose relevant files changed.

## Derive new VASP input

{py:class}`~matrifflow.pymatgen.io.vasp.sets.MRFDerivedInputSet`
inherits the inputs and final structure from an existing calculation,
changes them, and writes a *new* directory:

```python
from matrifflow.pymatgen.io.vasp.sets import MRFDerivedInputSet
from matrifflow.pymatgen.io.vasp.vaspdir import MRFVaspDir

source = MRFVaspDir('relaxed_calculation')
print(source['INCAR'])
print(source.final_structure)
if source.final_structure:
    print("Using final structure")
else:
    print("Falling back to POSCAR")

inputs = MRFDerivedInputSet(
    directory=source,
    functional='optB88-vdW',
    user_incar_settings={'NSW': 500, 'IBRION': 2},
)
inputs.structure = inputs.structure.copy()
inputs.structure.translate_sites([0], [0, 0, 0.02], frac_coords=True)
inputs.write_input('displaced_calculation')
```

`MRFDerivedInputSet` inherits the source settings and writes new
modified VASP inputs to `displaced_calculation`, and logs settings to
`MRFVaspInputSet.log`, and `INCAR.toml`.

## ATAT structure files

```python
from matrifflow.pymatgen.core.structure import MRFStructure

s = MRFStructure.from_file("str.out")  # vacancies become "X"
```

{py:class}`~matrifflow.pymatgen.core.structure.MRFStructure` is a
`pymatgen.core.Structure` that also reads and writes ATAT `str.out`
files.

## Transformations

### Inserting molecules

```python
from matrifflow.pymatgen.transformations.insert_molecule import InsertMoleculeTransformation

transformer = InsertMoleculeTransformation(
    molecule="Li",
    step=0.2,
    proximity_threshold=0.75,
)

inserts = transformer.all_inserts("host_structure.cif")
```

{py:class}`~matrifflow.pymatgen.transformations.insert_molecule.InsertMoleculeTransformation`
documents the grid parameters (`step`, `step_noise`, `anglestep`) and the
filtering options;

### Symmetry cloning

Given a `host` structure and a `defect` structure, generate every
symmetry-equivalent placement of an interstitial under the host's
space group.

This is used by [mrf derive . fill](cli.md#mrf-derive).

```python
from matrifflow.pymatgen.transformations.symmetry_clone import SymmetryCloneTransformation
from pymatgen.core import Structure

host = Structure.from_file("host.cif")
defect = Structure.from_file("host_with_interstitial.cif")

trans = SymmetryCloneTransformation(sym_operations=host)
clones = trans.get_all_clones(host_with_interstitial)
```

{py:class}`~matrifflow.pymatgen.transformations.symmetry_clone.SymmetryCloneTransformation`
takes the symmetry operations of the host.

## Diffusion analysis

```python
from matrifflow.pymatgen.diffusion.neb import get_neb_pairs

pairs = get_neb_pairs(
    structures=stable_sites_list,
    prototype=host_structure,
    cutoff="auto",
    remove_compound=True,
)

for start, end in pairs:
    print(f"Path from {start} to {end}")
```

{py:func}`~matrifflow.pymatgen.diffusion.neb.get_neb_pairs` generates and
filters the candidate pairs; the graph behind the diffusion paths, with
its distance and barrier helpers, is
{py:class}`~matrifflow.pymatgen.diffusion.neb.NEB_Graph`.
