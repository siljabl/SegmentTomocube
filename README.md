# Segmenting monolayers from 3D QPI
Code to extract 2D height and refractive index fields from 3D QPI images of epithelial monolayers. 

## code:
- `bioMlM_predictions.py`: Uses ML model to predict the probability that a voxel is part of a cell. Taken from [HTH-TomocubeAnalysis](https://github.uio.no/Hybrid-Technology-Hub/HTH-TomocubeAnalysis).
- `segment_3D_images.py`: Segments monolayer based on ML probabilities.
- `transform_3D_to_2D.py`: Turns 3D masks into 2D images of heights (in 1000*um) and refractive indices.
- `blur_and_rescale_intensity.ipynb`: Applying Gaussian blur to 3D data and adjusting height evolution to polynomial. To be merged with 'transform_3D_to_2D.py'.
- `src/`: Source code taken from [HTH-TomocubeAnalysis](https://github.uio.no/Hybrid-Technology-Hub/HTH-TomocubeAnalysis).
- `utils/`: Source code specific to this project.
- `notebooks/`: Notebooks for testing and tuning preprocessing parameters.

