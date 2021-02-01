# Survey on Implicit Neural Representations

[Chajin Shin](https://github.com/ChajinShin)<sup>\*1</sup> [Taeoh Kim](https://taeoh-kim.github.io/)<sup>\*1</sup>
<sup>*</sup> indicates equal contribution<br>
<sup>1</sup> Yonsei University<br>

#### Update log

*2020/01/17* - Initialize Repository<br>
*2020/02/01* - Public Release


## Summary Table

|    Name     |     Field     |                   Task                     | Published In |   Uploaded In  |
|:-----------:|:-------------:|:------------------------------------------:|:------------:|:--------------:|
|Dynamic NeRF |   4D Vision   |          4D Rendering                      |       -      |  Arxiv'201205  |
|     NeRF    |   3D Vision   |          View Synthesis                    |    ECCV'20   |  Arxiv'200319  |
|     MNSR    |   3D Vision   |          View Synthesis                    |  NeurIPS'20  |  Arxiv'200322  |
|    SINSR    |   3D Vision   |          View Synthesis                    |      3DV     |  Arxiv'200328  |
|     NSVF    |   3D Vision   |          View Synthesis                    |  NeurIPS'20  |  Arxiv'200722  |
|    NRFAA    |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'200809  |
|    NeRF++   |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201015  |
|    D-NeRF   |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201125  |
|     DeRF    |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201125  |
|   AutoInt   |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201203  |
|     NeRD    |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201207  |
|     NeRV    |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201207  |
|  DI-Fusion  |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201210  |
|    NRNeRF   |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'201222  |
|     PVA     |   3D Vision   |          View Synthesis                    |       -      |  Arxiv'210107  |
|   IF-Nets   |   3D Vision   |          3D Rendering                      |    ECCV'20   |  Arxiv'200920  |
|     GRF     |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201009  |
|   SDF-SRN   |   3D Vision   |          3D Rendering                      |  NeurIPS'20  |  Arxiv'201020  |
|     NDF     |   3D Vision   |          3D Rendering                      |  NeurIPS'20  |  Arxiv'201026  |
|    i3DMM    |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201128  |
|     NDG     |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201202  |
|    PNeRF    |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201210  |
|     DOP     |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201214  |
|     LCRF    |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201217  |
|  Vid2Actor  |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201223  |
|  NeuralBody |   3D Vision   |          3D Rendering                      |       -      |  Arxiv'201231  |
|    LISLF    |   3D Vision   |          3D Surface                        |    3DV'20    |  Arxiv'200327  |
|  Iso-Points |   3D Vision   |          3D Surface                        |       -      |  Arxiv'201211  |
|     NLF     |   3D Vision   |          3D Printing                       |    ACM'20    |    ACM'20      |
|    iNeRF    |   3D Vision   |          Camera Pose Estimation            |       -      |  Arxiv'201210  |
|NLOS via NeTF|   3D Vision   |          Non-line-of-Sight Imaging         |       -      |  Arxiv'210102  |
|    STNIF    |   2.5D Vision |          Space-Time View Synthesis         |       -      |  Arxiv'201125  |
|     NSFF    |   2.5D Vision |          Space-Time View Synthesis         |       -      |  Arxiv'201126  |
|     STaR    |   2.5D Vision |          Space-Time View Synthesis         |       -      |  Arxiv'210122  |
| spatial-VAE |   2D Vision   |          Image Deep Representation         |  NeurIPS'19  |  Arxiv'190925  |
|     HFIR    |   2D Vision   |          Image Super-Resolution            |   ICANN'19   |  Arxiv'190227  |
|     GRAF    |   2D Vision   |          Image Synthesis                   |       -      |  Arxiv'200705  |
|   X-Fields  |   2D Vision   |          Image Interpolation               |     TOG'20   |  Arxiv'201001  |
|     AGCI    |   2D Vision   |          Unconditional Image Generation    |       -      |  Arxiv'201124  |
|   GIRAFFE   |   2D Vision   |          Image Synthesis                   |       -      |  Arxiv'201124  |
|     CIPS    |   2D Vision   |          Unconditional Image Generation    |       -      |  Arxiv'201127  |
|   ASAP-net  |   2D Vision   |          Image Translation                 |       -      |  Arxiv'201205  |
|     LIIF    |   2D Vision   |          Image Super-Resolution            |       -      |  Arxiv'201216  |
|    SIREN    |   Vision      |          Others                            |  NeurIPS'20  |  Arxiv'200617  |
|     FFM     |   Vision      |          Others                            |       -      |  Arxiv'200618  |




## 4D Rendering
- **[Dynamic NeRF]** Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction  | **[Arxiv'201205]** |[`[pdf]`](https://arxiv.org/abs/2012.03065) [`[official code]`](https://github.com/gafniguy/4D-Facial-Avatars) 
    - They reconstruct 4D facial avatar neural radiance field from a short monocular portrait video sequence to synthesize novel head poses and changes in facial expression. This model need a portrait video and an image with only background as a inputs. Using 3D morphable model, they apply facial expression tracking. Then dynamic radiance field network takes position, view, facial expression, and per-frame learnable latent code as a inputs then predicts RGB value and volume density.



## View Synthesis

- **[NeRF]** NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | **[ECCV' 20]** | **[Arxiv'200319]** |[`[pdf]`](https://arxiv.org/abs/2003.08934) [`[official code - tensorflow]`](https://github.com/bmild/nerf) 
    - Synthesizing novel views of complex scene using continuous 5D coordinates (x, y, z, &theta;, &phi; ) to predict output of the volume density &sigma; and view-dependent emitted color (r,g,b).
  

- **[MNSR]** Multiview Neural Surface Reconstruction by Disentangling Geometry and Appearance | **[NeurIPS' 20]** | **[Arxiv'200322]** |[`[pdf]`](https://arxiv.org/abs/2003.09852) [`[official code - tensorflow]`](https://github.com/lioryariv/idr) 
    - Proposing end-to-end scheme that learn 3D geometry, appearance, and precise camera from masked 2D images and rough camera estimates. Outputs of network are RGB and mask values.
  

- **[SINSR]** Semantic Implicit Neural Scene Representations With Semi-Supervised Training  | **[3DV'20]** | **[Arxiv'200328]** |[`[pdf]`](https://arxiv.org/abs/2003.12673)
    - Existing implicit representation is used to estimate only appearance and 3D geometry information in a scene. They show that implicit representation can perform  per-point semantic segmentation while retaining its ability to represent appearance and geometry. They pretrain the network for novel view synthesis of RGB images. Then freeze it and train segmentation network. At test time, network infer the latent code of the novel object from single posed RGB image. Then render multi-view of novel RGB scenes and semantic segmentation views.


- **[NSVF]** Neural Sparse Voxel Fields | **[Arxiv'200722]** |[`[pdf]`](https://arxiv.org/abs/2007.11571) [`[official code - pytorch]`](https://github.com/facebookresearch/NSVF) 
    - Instead of modeling the entire space with a single implicit function, they assign a voxel embedding at each vertex of the voxel to obtain the representation of a query point from only a set of posed 2D images. Then they pass through an MLP to predict geometry and appearance of that query point. 


- **[NRFAA]** Neural Reflectance Fields for Appearance Acquisition | **[Arxiv'200809]** |[`[pdf]`](https://arxiv.org/abs/2008.03824)
    - Proposing a novel neural reflectance field representation that predicts not only geometry of scene but also reflectance of light. Network takes light and camera view, and coordinates then predicts volume density, normal vector, and reflectance properties.


- **[NeRF++]** NeRF++: Analyzing and Improving Neural Radiance Fields | **[Arxiv'201015]** |[`[pdf]`](https://arxiv.org/abs/2010.07492) [`[official code - pytorch]`](https://github.com/Kai-46/nerfplusplus) 
    - NeRF has a potential failure modes. Shape-radiance ambiguity show that inherent ambiguity between 3D shape and radiance that can admit degenerate solutions. NeRF use MLP structure that implicitly encodes a smooth BRDF prior on surface reflectance. Inverted sphere parameterization is related to dynamic range of the true scene depth. For background and foreground object, we need sufficient resolution each. They treat scene space with two separate volume to solve this problem.


- **[D-NeRF]** Deformable Neural Radiance Fields | **[Arxiv'201125]** |[`[pdf]`](https://arxiv.org/abs/2011.12948)
    - Proposing a method for photorealistic reconstruction of a non-rigidly deforming scene using photos or videos captured casually from mobile phones. Using coordinates from observation frame and deformation code, network predict a coarse canonical frame. With coordinates from a canonical frame and appearance code, refinement network predicts RGB color and density.


- **[DeRF]** DeRF: Decomposed Radiance Fields | **[Arxiv'201125]** |[`[pdf]`](https://arxiv.org/abs/2011.12490)
    - Using spatially decomposition of a scene and dedicating NeRF networks for each decomposed part to accel inference time. This network is 3 time efficient than NeRF with the same rendering quality. The final color value for a ray is calculated from segment of radiance and density, and alpha compositing.


- **[AutoInt]** AutoInt: Automatic Integration for Fast Neural Volume Rendering  | **[Arxiv'201203]** |[`[pdf]`](https://arxiv.org/abs/2012.01714)
    - Using radiance field for view synthesis has an extreme computational complexity and memory. To cope this problem, they instantiate the computational graph corresponding to the derivative of the implicit neural representation. Then they build the grad network and optimize it.


- **[NeRD]** NeRD: Neural Reflectance Decomposition from Image Collections  | **[Arxiv'2012107]** |[`[pdf]`](https://arxiv.org/abs/2012.03918) [`[official code]`](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition)
    - When illumination is not a single light source, decomposing a scene into shape, reflectance, and illumination is a very challenging problem. Using neural radiance field, they decompose a scene and generate novel views under any illumination in real-time. Input is a set of images of an object and known camera pose. Specifically, network takes ray with sampling position and spherical gaussian parameter per image and predict output color.


- **[NeRV]** NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis  | **[Arxiv'201207]** |[`[pdf]`](https://arxiv.org/abs/2012.03927)
    - This model takes a set of images of an object views illuminated by unconstrained known lighting and predicts 3D representation from novel viewpoints with continuous volumetric function. They propose neural visibility fields to reduce the computational complexity of volume rendering a camera ray.


- **[DI-Fusion]** DI-Fusion: Online Implicit 3D Reconstruction with Deep Priors  | **[Arxiv'201203]** |[`[pdf]`](https://arxiv.org/abs/2012.05551)
    - Existing 3D reconstruction need massive memory and low surface quality. They reconstruct 3D-scene using implicit representation to cope this problem. Network first predict camera pose from input with encoding scheme. Then RGB and density are predicted at decoder.


- **[NRNeRF]** Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Deforming Scene from Monocular Video  | **[Arxiv'201222]** |[`[pdf]`](https://arxiv.org/abs/2012.12247) [`[official code - pytorch]`](https://github.com/facebookresearch/nonrigid_nerf)
    - They propose the model that reconstruct novel free viewpoint non-rigid scenes via ray blending using only monocular video of the deforming scene. Network takes RGB images and camera parameters then predicts per-time-step space geometry and appearance representations. 


- **[PVA]** Pixel-aligned Volumetric Avatars | **[Arxiv'210107]** |[`[pdf]`](https://arxiv.org/abs/2011.12490)
    - Volumetric models represent fine structure such as hair, but cannot be extended to the multi-identity setting. This paper devise a novel approach for predicting volumetric models of the human head with a few example views. Given a target viewpoint, network extract local, pixel-aligned features. Then neural radiance field takes the features to predict RGB and density value.





## 3D Rendering

- **[IF-Nets]** Implicit Feature Networks for Texture Completion from Partial 3D Data | **[Arxiv'200920]** |[`[pdf]`](https://arxiv.org/abs/2009.09458)  [`[official code - pytorch]`](https://github.com/jchibane/if-net) 
    - They focus on 3D texture and geometry completion from incomplete 3D scans. They won the SHARP ECCV'20 challenge. IF-Nets in-paints the missing texture parts using 3D geometry and 3D partial texture.


- **[GRF]** GRF: Learning a General Radiance Field for 3D Scene Representation and Rendering  | **[Arxiv'201009]** |[`[pdf]`](https://arxiv.org/abs/2010.04595)
    - Rendering arbitrary complex 3D scenes using 2D observations. Main idea is that explicitly using multi-view geometry method to obtain representation from several observed 2D views. The network takes images, camera poses, and intrinsics as an input and predicts RGB values and volume density.


- **[SDF-SRN]** SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images  | **[NeurIPS' 20]** | **[Arxiv'201020]** |[`[pdf]`](https://arxiv.org/abs/2010.10505)  [`[official code - pytorch]`](https://github.com/chenhsuanlin/signed-distance-SRN) 
    - This model requires only a single view of objects at training time to reconstruct 3D objects. They derive new differentiable rendering formulation for learning signed distance functions from 2D silhouettes. Hypernetwork encoder takes image and predicts parameters of implicit function. Implicit function predicts RGB and signed distance value from 3D coordinates.


- **[NDF]** Neural Unsigned Distance Fields for Implicit Function Learning | **[NeurIPS' 20]** | **[Arxiv'201026]** |[`[pdf]`](https://arxiv.org/abs/2007.11571) [`[official code - pytorch]`](https://github.com/jchibane/ndf) 
    - Implicit neural representation can only adapt closed surfaces, so they propose the unsigned distance field for arbitrary 3D shapes given sparse point clouds. 
  

- **[i3DMM]** i3DMM: Deep Implicit 3D Morphable Model of Human Heads  | **[Arxiv'201128]** |[`[pdf]`](https://arxiv.org/abs/2011.14143)
    - They propose novel morphable model Synthesizing full 3D face including hair. Unlike to mesh-based models, this model is trained on rigidly aligned scans and semantically disentangle the geometry and color components. Shape deformation network takes coordinates and latent code. Reference shape network and color network take output of DeformNet and latent code color to predict color values and signed distance function.
  

- **[NDG]** Neural Deformation Graphs for Globally-consistent Non-rigid Reconstruction  | **[Arxiv'201202]** |[`[pdf]`](https://arxiv.org/abs/2012.01451)
    - They model a per-frame viewpoint and surface consistent deformation graph network for non-rigid object. They use implicit representation for optimizing geometry of the object. Neural deformation graph takes graph node position, rotation, and weights. Using sample point (x, y, z), and predicts signed distance function values.


- **[PNeRF]** Portrait Neural Radiance Fields from a Single Image | **[Arxiv'201210]** |[`[pdf]`](https://arxiv.org/abs/2012.05903)
    - NeRF requires multiple images to train. This model pre-trains the weights of an MLP and uses meta-learning to predict volumetric density and colors with only a single headshot portrait. They provide a multi-view portrait dataset consisting of controlled captures in a light stage.
  

- **[DOP]** Deep Optimized Priors for 3D Shape Modeling and Reconstruction  | **[Arxiv'201214]** |[`[pdf]`](https://arxiv.org/abs/2012.07241)
    - Existing approach has a difficulty in generalizing for unseen data. They combine learning-based and optimization-based methods. They did not fix the pretrained prior at test time. They optimize the pre-trained prior and latent code using task-specific dataset with neural implicit field. Here, neural implicit field takes 3D coordinates and latent code as inputs, then predicts signed distance field.


- **[LCRF]** Learning Compositional Radiance Fields of Dynamic Human Heads  | **[Arxiv'201217]** |[`[pdf]`](https://arxiv.org/abs/2012.09955)
    - Proposing a novel compositional 3D representation that combines previous encoder-decoder scheme and radiance fields method so that they get fine detail high resolution and fast results. Network takes multi-view video as input and predicts coarse 3D local animation codes. Using this codes, view vector, and coordinates, radiance function of MLP predicts output color and differential probability of opacity.


- **[Vid2Actor]** Vid2Actor: Free-viewpoint Animatable Person Synthesis from Video in the Wild | **[Arxiv'201223]** |[`[pdf]`](https://arxiv.org/abs/2012.12884)
    - They reconstruct an animatable model of the person from the given video of a person without explicit 3D mesh reconstruction. Network takes input video frames of a human subject and recover 3D body pose and camera parameters. Then using results of pose and maps voxels, network renders to RGB&alpha;.


- **[NeuralBody]** Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans  | **[Arxiv'201217]** |[`[pdf]`](https://arxiv.org/abs/2012.15838) [`[official code]`](https://github.com/zju3dv/neuralbody)
    - For human body representation, if the given views are highly sparse, then task will be ill-posed problem. To cope it, network learn neural representation from different frames share the same set of latent codes so that the network can integrate observations across the frame. Using this method, the network can synthesize photorealistic novel views of human body with complex motions from a sparse multi-view video. Network takes structured latent codes and predicts latent code volume. Using this code, coordinates, and camera directions, it predicts color and density.





## 3D Surface

- **[LISLF]** Learning Implicit Surface Light Fields  | **[3DV'20]** | **[Arxiv'200327]** |[`[pdf]`](https://arxiv.org/abs/2003.12406)
    - While existing method use simple texture models, they propose novel implicit representation for capturing the visual appearance of object surface. With RGB image and corresponding 3D object, this model synthesize new view. Network takes location and color of point light source, object shape and image content encoded vector, and view direction as inputs and predicts RGB values.


- **[Iso-Points]** Iso-Points: Optimizing Neural Implicit Surfaces with Hybrid Representations  | **[Arxiv'201211]** |[`[pdf]`](https://arxiv.org/abs/2012.06434)
    - Neural implicit functions show powerful representation ability for surfaces in 3D. However, still accurate and robust reconstruction is challenging problem. This network combine geometry-aware sampling and reconstruction to improve the fidelity. This model extract the iso-points via projection and resample then reconstruct surface.





## 3D Printing

- **[NLF]** Neural light field 3D printing  | **[ACM'20]** |[`[pdf]`](https://dl.acm.org/doi/pdf/10.1145/3414685.3417879) [`[official code]`](https://quan-zheng.github.io/publication/neuralLF3Dprinting20/)
    - Modern 3D printers still have challenging task in optimizing a displays in full 3D volume for a given light-field imagery. They propose novel method that encodes input light field imagery as a continuous space implicit representation. Network takes a position, and 3D direction then predicts the absorption coefficient.







## Camera Pose Estimation 

- **[iNeRF]** iNeRF: Inverting Neural Radiance Fields for Pose Estimation  | **[Arxiv'201210]** |[`[pdf]`](https://arxiv.org/abs/2012.05877)
    - Using pretrained NeRF, this model estimate camera pose. From the initial camera pose, first they sample pixels, not to render all points. Through a pre-trained NeRF, they get rendered pixels. Then they compare rendered pixels with given observed pixels to get difference L2 loss. Using backpropagation with several iteration, they get final estimated camera pose.








## Non-line-of-Sight Imaging

- **[NLOS via NeTF]** Non-line-of-Sight Imaging via Neural Transient Fields  | **[Arxiv'210102]** |[`[pdf]`](https://arxiv.org/abs/2101.00373)
    - Previous NLOS imaging use 3D geometry or voxel density. In this paper, they use spherical volume neural transient field. Network takes detection spot (x, y) on the wall, and scene point (r, &theta;, &phi;) of spherical coordinates then predicts volume density and albedo.




## Video Synthesis

- **[STNIF]** Space-time Neural Irradiance Fields for Free-Viewpoint Video | **[Arxiv'201125]** |[`[pdf]`](https://arxiv.org/abs/2011.12950)
    - Video contains only one observation of the scene at any point in time so that learning a spatiotemporal irradiance field is very challenges. They use scene depth to constrain the time-varying geometry of dynamic scene representation. Network takes coordinates and time as a input and predicts RGB color value and volume density.


- **[NSFF]** Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes | **[Arxiv'201126]** |[`[pdf]`](https://arxiv.org/abs/2011.13084)
    - Synthesizing novel views with times for dynamic scenes using monocular video and camera position information. The network takes a position and viewing direction and produces RGB value and volumetric density. They train a standard IF-Net to reconstruct untextured geometry given the corrupted input and then predicts RGB value at point p.
    

- **[STaR]** STaR: Self-supervised Tracking and Reconstruction of Rigid Objects in Motion with Neural Rendering | **[Arxiv'210122]** |[`[pdf]`](https://arxiv.org/abs/2101.01602)
    - They propose novel method that reconstruct dynamic scenes with rigid motion from multi-view RGB videos without any manual annotation. Existing methods show poor reconstruction ability when object in the scene move. They sample points on the original ray then pass to the neural networks with coordinates to predict color.






## Image Synthesis

- **[spatial-VAE]** Explicitly disentangling image content from translation and rotation with spatial-VAE | **[NeurIPS' 19']** | **[Arxiv'190925']**  |[`[pdf]`](https://arxiv.org/abs/1909.11663) [`[official code - pytorch]`](https://github.com/tbepler/spatial-VAE)
    - Explicitly separating rotation and translation generative factors from images using continuous generative factors. The network takes pixel coordinates with latent variables and predicts output pixel RGB value. Then inference network predicts generative factors of latent variables, rotation and translation factors. Trained on MNIST dataset.


- **[HFIR]** Hypernetwork Functional Image Representation | **[ICANN'19]** | **[Arxiv'201216]** |[`[pdf]`](https://arxiv.org/abs/1902.10404)
    - Target network takes coordinates and predicts output pixel RGB value. Hypernetwork takes input image to generate parameters of a target network. With this scheme, this model can learn continuous representation function of images. Trained on Set5, Set14, B100, Urban100 dataset.


- **[GRAF]** GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis | **[Arxiv'200705]**  |[`[pdf]`](https://arxiv.org/abs/2007.02442) [`[official code - pytorch]`](https://github.com/autonomousvision/graf)
    - They use neural radiance fields for high resolution 3D-aware image synthesis. To generate varying view images, generator takes camera matrix, camera pose, 2D sampling pattern, and appearance codes and predicts an image patch. They release their own synthesized dataset.


- **[X-Fields]** X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation | **[TOG' 20]** | **[Arxiv'201001]** |[`[pdf]`](https://arxiv.org/abs/2010.00450) [`[official code - tensorflow]`](https://github.com/m-bemana/xfields)
    - Proposing X-Field to generate 2D image set of different view, time, or illumination conditions. Network takes new coordinates of view, time, and light for processing interpolation to make unobserved images. They split images into shade and albedo, then interpolate and warp using observed images to generate unobserved coordinates images.


- **[AGCI]** Adversarial Generation of Continuous Images | **[Arxiv'201124]**  |[`[pdf]`](https://arxiv.org/abs/2011.12026) [`[official code - pytorch]`](https://github.com/universome/inr-gan)
    - Generating unconditional images using implicit neural representation with an adversarial scheme. To overcome unstable training and long test time, they used FMM, multi-scale INRs. Trained on LSUN-bedroom, FFHQ.


- **[GIRAFFE]** GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields | **[Arxiv'201124]**  |[`[pdf]`](https://arxiv.org/abs/2011.12100)
    - Incorporating a compositional 3D scene representation into the generative model so that it leads to more controllable image synthesis for objects. They disentangle not only shapes and appearance of objects but also objects from the background. Generator takes a camera pose, N appearance codes, and affine transformation. N codes are consist of N-1 objects and 1 background. Then it predicts an image using predicted 3D points.
  

- **[CIPS]** Image Generators with Conditionally-Independent Pixel Synthesis | **[Arxiv'201127]** |[`[pdf]`](https://arxiv.org/abs/2011.13775)
    - Generating unconditional images without adversarial learning. They use pixel coordinates (x, y) with latent vector to get RGB value of output pixel. Network architecture is based on StyleGAN2. Trained on LSUN-Churches or FFHQ.
    

- **[ASAP-net]** Spatially-Adaptive Pixelwise Networks for Fast Image Translation | **[Arxiv'201205']**  |[`[pdf]`](https://arxiv.org/abs/2012.02992)
    - Proposing fast and efficient scheme for high resolution image translation using pixel-wise network. For high frequency details, they use low-resolution hypernetwork and coordinate positional encoding.
   

- **[LIIF]** Learning Continuous Image Representation with Local Implicit Image Function | **[Arxiv'201216]** |[`[pdf]`](https://arxiv.org/abs/2012.09161) [`[official code - pytorch]`](https://github.com/yinboc/liif) 
    - Learning continuous-scale image representation for single image super-resolution. The network takes pixel coordinates (x, y) with neighborhood features and predicts output pixel RGB. Trained on DIV2K.






## Others

- **[SIREN]** Implicit Neural Representation with Periodic Activation Functions | **[NeurIPS' 20]**  |  **[Arxiv'200617]** |[`[pdf]`](https://arxiv.org/abs/2006.09661)  [`[official code - pytorch]`](https://github.com/vsitzmann/siren)
    - To enhance fine detail with implicit neural representation, they propose new periodic activation function: *SIREN* using sine function. They also propose a new initialization scheme for *SIREN*.


- **[Fourier Feature Mapping]** Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains | **[Arxiv200618]**  |[`[pdf]`](https://arxiv.org/abs/2006.10739)  [`[official code]`](https://github.com/tancik/fourier-feature-networks)
    - For regression problem in 2D-images or 3D-objects or scene, coordinate-based MLP have difficulty with learning high frequency details due to 'spectral bias'. They use a fourier feature mapping to transform the effective neural tangent kernel into a stationary kernel with a tunable bandwidth.


