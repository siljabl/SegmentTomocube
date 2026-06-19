# Segmenting monolayers from 3D QPI
Code to extract 2D height and refractive index fields from 3D QPI images of epithelial monolayers. 

![illustration of segmentation pipeline](figs/illustration.svg)


## Used in:
- _Quantitative Phase Imaging of Epithelial Monolayer Dynamics_, [Bioarxiv](https://www.biorxiv.org/content/10.64898/2026.01.17.700037v1)
- _Three Dimensional Dynamics of Epithelial Monolayers_, [Bioarxiv](https://www.biorxiv.org/content/10.64898/2026.03.10.710903v1.abstract)

## Scripts:
- `bioMlM_predictions.py`: Uses ML model to predict the probability that a voxel is part of a cell. Taken from [HTH-TomocubeAnalysis](https://github.uio.no/Hybrid-Technology-Hub/HTH-TomocubeAnalysis).
- `segment_3D_stacks.py`: Segments monolayer based on ML probabilities.
- `transform_3D_to_2D.py`: Turns 3D masks into 2D images of heights (in units of 0.1 µm to match HoloMonitor format) and refractive indices.
- `src/`: Source code. Partly taken from [HTH-TomocubeAnalysis](https://github.uio.no/Hybrid-Technology-Hub/HTH-TomocubeAnalysis).
- `notebooks/`: Notebooks for testing and tuning preprocessing parameters.

