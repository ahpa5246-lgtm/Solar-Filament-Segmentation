# Third-party source and licenses

## V5 reference: Mask R-CNN + U-Net Refiner

- **Original author:** Nomannic (Kaggle).
- **Original title:** [CV] Solar Filament | Mask R-CNN + U-Net Refiner.
- **Original source:** https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner
- **Original license:** Apache License, Version 2.0. License text: https://www.apache.org/licenses/LICENSE-2.0.txt
- **Our changes:** Build only the training/submission cells, add two-GPU CUDA preflight, add 4-view flip TTA as a validation-selected ablation, verify all test IDs and COCO RLE. These modifications are clearly labelled V5 in the generated notebook.
- **Distribution:** The original notebook is fetched at build time using the public Kaggle API. Recovered project derivatives are preserved in `notebooks/baselines/v5_version5.ipynb` and `v5_training_session_v3.ipynb`. The original Apache 2.0 license and upstream authorship continue to apply to adapted material. A copy of that license is included in `third_party/Apache-2.0.txt`.
- **V6 changes:** `solar/v6_refiner_audit.py` extracts the exact archived `ConvBlock`, `CropUNet`, `square_bounds`, and refiner prediction helpers from the fingerprinted project derivative. Its surrounding evaluation, threshold search, confidence NMS and resource guards are new project code. No upstream training cells execute during extraction.

V5 achieved owner-confirmed official Public PQ **0.32**. This is our project result, not a claimed reproduction of another team's leaderboard score. The recovered API training session's association to that submission is not independently verified.

## Additional research and libraries
- **torchvision**: PyTorch Vision detection models, https://github.com/pytorch/vision, BSD-style license (see upstream).
- **EdgeAttNet**: Victor Solomon et al., https://github.com/dasjar/EdgeAttNet, MIT license; V4 uses an independently implemented *inspired* bottleneck, not their code.
- **soft-clDice**: Shit et al., CVPR 2021, https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html.

Only official competition training annotations are used as supervised labels; publicly available competition test labels are prohibited for training.
