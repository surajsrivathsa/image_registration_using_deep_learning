# Deformable Image Registration using Deep Learning
University project - Deep learning based Image Registration Techniques

Description: This project is split into two phases.
Phase 1: Comparitive study of SOTA Deformable Intra and Intermodal Image Registration Techniques vs Traditional Methods (Chatterjee et al., [2020](https://www.researchgate.net/publication/349588959_A_Comparative_Study_of_Deep_Learning_Based_Deformable_Image_Registration_Techniques))

Deep Learning based methods
- ADMIR (Tang et al., [2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9062524))
- Voxelmorph (Balakrishnan et al., [2018](https://ieeexplore.ieee.org/document/8633930))
- FIRE (Wang et al., [2019](https://arxiv.org/pdf/1907.05062.pdf))
- ICNET (Zhang et al., [2018](https://arxiv.org/pdf/1809.03443.pdf))
  
Traditional Methods
- ANTS ([Avants et al.](http://stnava.github.io/ANTs/))
- FSL (Jenkinson et al., [2002](https://pubmed.ncbi.nlm.nih.gov/12377157/))

Phase 2a: Proposal of new deep learning based method MSCGUNet (Multiscale Self Constructing Graph UNet)
- Multiscale Image input to handle different amounts of deformations easily (Chatterjee et al., [2020](https://arxiv.org/pdf/2006.10802.pdf))
- SCG Net (Liu et al., [2020](https://arxiv.org/pdf/2003.06932.pdf)) to self construct graph from encoder and handle semantics about brain.
- Cycle Consistency to regularize and make deformation field consistent

Phase 2b: Proposal of direct optimization of deformation field using gradient descent without any network
- How far will we get with simple gradient descent on deformation field directly and without using other networks?
- Tried with different optimizers and epochs
- Surprisingly good results and performed close to ANTs and other Deep learning based models