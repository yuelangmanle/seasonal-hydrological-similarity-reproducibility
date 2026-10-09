# Verification record

The repository was checked from a fresh extracted copy with Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.1, and Pillow 12.3.0. The installed versions are listed in `requirements-tested.txt`.

The verification run completed the following checks:

- Python compilation for all repository scripts;
- analytical representation checks in `reproduction/test_seasonal_transport.py`;
- primary replay and held-out score comparison;
- benchmark controls and sensitivity checks;
- climate-stratified score replay;
- temporal-transfer and prediction-control replays from cached inputs;
- figure redraw and width exports;
- checksum verification for the distributed source and result files.

The primary replay reproduced the archived monthly conditional score (`0.2904733733`) and annual FDC score (`0.4528633063`). Figure source tables matched their archived numerical values to `1e-8` in the tested environment.

Rendering output can differ slightly between operating systems and graphics-library versions because of font discovery, antialiasing, and raster export. Numerical checks use the tolerances recorded in the individual protocols.
