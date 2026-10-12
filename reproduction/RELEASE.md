# Plotting-data release 2026-10-12

Download the [2026-10-12 plotting-data ZIP](https://drive.google.com/file/d/1WbO4nQGvkneY76LWbJ3rtrBpHijMbKAV/view?usp=sharing) from Google Drive.
Link access is read-only; the historical archive remains available separately.

Archive: `simesh-figure-data-2026-10-12.zip`

Size: 91,599,157 bytes

SHA-256: `1ad92d62d05288dec1227d3c9d035f883a5e5d04c75c53c60653478ec12f9141`

The ZIP contains a top-level `data/` directory with 41 manifest-verified files
and the manifest itself. Extract it into `reproduction/` and run `verify.py`
and `render.py` from the repository root. Use the matching script tag
`paper-figures-2026-10-12`, and retain `git rev-parse HEAD` with outputs.

This release adds the recorded performance comparison, updates the thermal
products to absorbed synthetic emission and emission-measure diagnostics, and
uses current manuscript numbering. Framework, architecture, and AMR layout
updates are in the scripts. Representation profiles, current-sheet diagnostics,
and filament topology retain their verified numerical products. No original
simulation snapshots are included. See `DATA.md` for units and provenance.

The historical 2026-09-21 bundle remains separate and uses Figures 1–7. Its
[Google Drive archive](https://drive.google.com/file/d/1XglvwIhu7mi8HKRKTBa28WoX4GM6d75G/view?usp=sharing)
has SHA-256
`8208bc79c354e551b9137df2cd9a31215aaef6e6223ac444715229daaadd8a1a`.
Its matching historical scripts are available at commit
[`5d586961`](https://github.com/Astery0502/simesh/tree/5d586961be7eace8000a1c7cfc1beb15c789fdcb/reproduction).
