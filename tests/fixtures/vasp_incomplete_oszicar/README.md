# Incomplete OSZICAR

Selected, unchanged lines from the Li/CNT calculation:

```
2026.CNT.pore-filling/04.generations.Li/optB88-vdW/
CNT_17,0_1_10.0.KPOINTS.1x1x8.PREC.Accurate.ENCUT.550,ISPIN.2/
3_Li/gen_0_idx_16/relax.2/OSZICAR
```

The excerpt retains the headers, first electronic iterations, and ionic
summaries of steps 419 and 420, then the header, first electronic iteration,
trial line, and final electronic iteration of the unfinished step.
Intermediate electronic iterations are omitted.

Pymatgen parses the trailing `trl` line as an empty ionic-step dictionary.
The latest recorded `E0` is -540.35014 eV (step 420), not the electronic
energy of the unfinished step. The two distinct ionic energies also test
that MatRiFFlow searches from the end without changing the step order.
