# image_registration_using_deep_learning
University project - Deep learning based Image Registration Techniques

Description: This project is split into two phases.
Phase 1: Comparitive study of SOTA Deformable Intra and Intermodal Image Registration Techniques vs Traditional Methods

Deep Learning based methods
- ADMIR
- Voxelmorph
- FIRE
- ICNET
  
Traditional Methods
- ANTS
- FSL

Phase 2a: Proposal of new deep learning based method MSCGUNet (Multiscale Self Constructing Graph UNet)
- Multiscale Image input to handle different amounts of deformations easily (Chatterjee et al., [2020](https://arxiv.org/pdf/2006.10802.pdf))
- SCG Net (Liu et al., [2020](https://arxiv.org/pdf/2003.06932.pdf)) to handle semantics about brain.
- Cycle Consistency to regularize and make deformation field consistent

Phase 2b: Proposal of direct optimization of deformation field using gradient descent without any network
- How far will we get with simple gradient descent on deformation field directly and without using other networks?
- Tried with different optimizers and epochs
- Surprisingly good results and performed close to ANTs and other Deep learning based models