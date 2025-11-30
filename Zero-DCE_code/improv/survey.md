# Milestone 3: Project Proposal - Low Light Image Enhancement Survey

**Authors:** Imrose Batterywala (314540010), Shahzeb Mohammed (314540021)  
**Course:** Image Processing, Fall 2025  
**Due Date:** 2025/12/3 23:59

---

## 1. Introduction

### 1.1 Problem Definition and Limitations

Low-light image enhancement is crucial for image understanding in poor lighting conditions. The baseline Zero-DCE (Zero-Reference Deep Curve Estimation) model, while widely used and practical, exhibits significant limitations when dealing with challenging scenarios:

1. **Extreme Low Light**: The model fails to adequately brighten extremely dark regions. Evaluation results show that approximately 33% of pixels remain below the darkness threshold (0.2) even after enhancement, with hard cases maintaining >90% dark pixel fraction.

2. **Uneven Illumination**: Zero-DCE struggles with non-uniform lighting conditions, leading to inconsistent enhancement across different image regions. The patch contrast increases 3× but often introduces harsh contrast that hides fine details.

3. **Overexposure**: The model tends to overexpose certain areas, particularly when attempting to brighten dark regions, resulting in loss of highlight details. Bright pixel fraction remains near zero (~2.4e-5), indicating failure to create usable highlights.

4. **Limited Aesthetic Improvement**: Despite significant tonal manipulations, the NIMA (Natural Image Assessment) score improves only marginally (+0.06 on average), suggesting minimal perceptual quality gain.

### 1.2 Baseline Methodology Overview

The Zero-DCE model reformulates enhancement as a deep curve estimation problem:

- **Network Architecture**: 7-layer CNN (DCE-Net) with 32 filters per layer (3×3), no pooling or batch normalization to preserve local contrast
- **Enhancement Mechanism**: Light-Enhancement Curve (LE-curve): LE(I) = I + αI(1 − I), iteratively applied 8 times
- **Training Losses**:
  - Spatial Consistency Loss: maintains local contrast stability
  - Exposure Control Loss: pushes brightness toward 0.6
  - Color Constancy Loss: prevents color cast
  - Illumination Smoothness Loss: smoothens curve maps

### 1.3 Proposed Methodology Overview

We propose to enhance Zero-DCE by integrating multiple complementary techniques:

1. **Hybrid Exposure Fusion**: Mix Zero-DCE output with synthetic exposure brackets to lift shadows without killing highlights
2. **Texture-Aware Lighting Maps**: Use gradient-respecting kernels to brighten dark regions without harsh contrast jumps
3. **Bright/Dark Balance Loss**: Add dual-histogram regularization to boost highlights and reduce deep-shadow persistence
4. **Perceptual Co-training**: Train with NIMA/NIQE heads to enhance both visibility and aesthetic realism

### 1.4 Data and Resources

**Datasets:**
- Existing test data: DICM (69 images), LIME (10 images)
- Additional test sets for extreme low-light scenarios
- Synthetic exposure brackets for fusion experiments

**Computational Resources:**
- PyTorch framework (existing implementation)
- Pre-trained Zero-DCE weights (Epoch99.pth)
- Evaluation metrics: NIMA, NIQE, PSNR, SSIM, patch contrast, dark/bright pixel fractions

**Additional Tools:**
- pyiqa library for NIMA/NIQE computation
- Image processing libraries (PIL, OpenCV, scikit-image)
- Visualization tools for comparative analysis

---

## 2. Approach

### 2.1 Methodologies and Strategies

#### 2.1.1 Hybrid Exposure Fusion

**Rationale**: Zero-DCE fails to create usable highlights while attempting to brighten dark regions. Exposure fusion can combine multiple exposure levels to preserve both shadow and highlight details.

**Implementation Steps**:
1. Generate synthetic exposure brackets from Zero-DCE enhanced image (e.g., -2EV, -1EV, 0EV, +1EV, +2EV)
2. Apply multi-scale exposure fusion algorithm to combine brackets
3. Integrate fusion result with Zero-DCE output using adaptive weighting based on local luminance
4. Preserve Zero-DCE's detail enhancement while recovering highlight information

**Expected Benefits**:
- Improved bright pixel fraction (target: >0.01)
- Better highlight preservation
- Reduced overexposure artifacts

#### 2.1.2 Texture-Aware Lighting Maps

**Rationale**: Current illumination smoothness loss may be too simplistic, leading to harsh contrast jumps. Gradient-respecting kernels can better preserve texture while enhancing illumination.

**Implementation Steps**:
1. Compute local gradient magnitude maps from input image
2. Design adaptive smoothing kernels that respect gradient boundaries
3. Modify illumination smoothness loss to incorporate gradient-aware regularization
4. Apply texture-preserving enhancement to dark regions

**Expected Benefits**:
- Reduced harsh contrast artifacts
- Better detail preservation in enhanced regions
- More natural-looking results

#### 2.1.3 Bright/Dark Balance Loss

**Rationale**: Current exposure control loss targets a single brightness level (0.6), which doesn't address the dual problem of persistent dark pixels and missing bright pixels.

**Implementation Steps**:
1. Compute dual histograms: one for dark regions (<0.2) and one for bright regions (>0.9)
2. Design loss terms:
   - Dark pixel reduction loss: penalize high fraction of pixels <0.2
   - Bright pixel promotion loss: encourage presence of pixels >0.9
3. Balance these losses with existing exposure control loss
4. Apply adaptive weighting based on input image characteristics

**Expected Benefits**:
- Reduced dark pixel fraction (target: <0.15 from current 0.316)
- Increased bright pixel fraction (target: >0.01 from current 2.4e-5)
- Better overall exposure balance

#### 2.1.4 Perceptual Co-training

**Rationale**: Current training focuses on low-level losses but doesn't explicitly optimize for perceptual quality. NIMA/NIQE-based losses can guide the model toward more aesthetically pleasing results.

**Implementation Steps**:
1. Integrate NIMA and NIQE as differentiable loss components
2. Design perceptual loss that combines:
   - NIMA score maximization (aesthetic quality)
   - NIQE score minimization (naturalness)
3. Balance perceptual losses with existing spatial/exposure/color losses
4. Fine-tune pre-trained Zero-DCE model with combined loss function

**Expected Benefits**:
- Improved NIMA scores (target: >4.15 from current 4.06)
- More natural-looking enhanced images
- Better subjective quality assessment

### 2.2 Expected Results

After implementing these improvements, we anticipate:

**Quantitative Improvements**:
- **Dark Pixel Fraction**: Reduction from 0.316 to <0.15 (50%+ improvement)
- **Bright Pixel Fraction**: Increase from 2.4e-5 to >0.01 (400× improvement)
- **NIMA Score**: Increase from 4.06 to >4.15 (significant aesthetic improvement)
- **Patch Contrast**: Maintain current levels (~0.080) while reducing harsh artifacts
- **Consistency**: More stable performance across extreme low-light, uneven illumination, and overexposure scenarios

**Qualitative Improvements**:
- Better visibility in extremely dark regions
- More natural highlight representation
- Reduced overexposure artifacts
- Improved overall aesthetic quality
- Better handling of uneven illumination

**Performance Characteristics**:
- Maintain Zero-DCE's lightweight and fast inference properties
- Minimal increase in computational overhead (<20%)
- Real-time processing capability preserved

---

## 3. Method Exploration

### 3.1 Survey of Related Works

#### 3.1.1 Zero-LEINR (Zero-Reference Low-light Image Enhancement with Intrinsic Noise Reduction)

**Authors**: W. H. Tang, H. Yuan, T.-H. Chiang, C.-C. Huang  
**Venue**: ISCAS 2023  
**DOI**: 10.1109/ISCAS46773.2023.10181743

**Method**: Integrates intrinsic noise reduction within the enhancement process without requiring additional denoising modules. The approach addresses noise amplification issues that commonly occur in low-light enhancement.

**Key Contributions**:
- Unified framework for enhancement and denoising
- No need for separate noise reduction pipeline
- Maintains zero-reference property

**Advantages**:
- Addresses noise amplification problem directly
- Efficient single-pass processing
- Preserves zero-reference training paradigm

**Disadvantages**:
- May not fully address extreme low-light scenarios
- Limited discussion on overexposure handling
- No explicit handling of uneven illumination

**Comparison with Our Approach**: Our approach addresses noise implicitly through texture-aware lighting maps and exposure fusion, but we could incorporate explicit noise reduction similar to Zero-LEINR. However, our focus is broader, targeting extreme low-light, uneven illumination, and overexposure simultaneously.

---

#### 3.1.2 BEENet (Balanced Enhancement Network for Uneven Exposure Low-Light Image Enhancement)

**Authors**: J. Zhang, Y. Zhou, E. Zhu, J. Sun  
**Venue**: ICFTIC 2024  
**DOI**: 10.1109/ICFTIC64248.2024.10913334

**Method**: Specifically designed for uneven exposure scenarios, using a balanced enhancement strategy that handles both overexposed and underexposed regions simultaneously.

**Key Contributions**:
- Explicit handling of uneven illumination
- Balanced enhancement across different exposure regions
- Improved performance on non-uniform lighting

**Advantages**:
- Directly addresses uneven illumination problem
- Balanced approach to over/under-exposure
- Recent work with good results

**Disadvantages**:
- May require more computational resources
- Less focus on extreme low-light scenarios
- Not zero-reference (may require training data)

**Comparison with Our Approach**: BEENet's balanced enhancement philosophy aligns with our bright/dark balance loss. However, our approach maintains zero-reference training while incorporating similar balanced enhancement concepts through loss function design.

---

#### 3.1.3 Residual Quotient Learning for Zero-Reference Low-Light Image Enhancement

**Authors**: C. Xie et al.  
**Venue**: IEEE Transactions on Image Processing, 2025  
**DOI**: 10.1109/TIP.2024.3519997

**Method**: Introduces residual quotient learning to improve zero-reference low-light enhancement, focusing on better feature representation and enhancement quality.

**Key Contributions**:
- Novel residual quotient learning framework
- Improved feature extraction for low-light images
- Better enhancement quality while maintaining zero-reference property

**Advantages**:
- Maintains zero-reference training
- Improved feature learning
- Better enhancement results

**Disadvantages**:
- More complex architecture
- May require longer training time
- Limited discussion on specific failure cases (extreme low-light, overexposure)

**Comparison with Our Approach**: While Residual Quotient Learning improves feature representation, our approach focuses on specific failure modes (extreme low-light, uneven illumination, overexposure) through targeted loss functions and fusion techniques. Our methods are complementary and could potentially be combined.

---

#### 3.1.4 Noise-aware Zero-Reference Low-light Image Enhancement for Object Detection

**Authors**: K. Ang, W. T. Lim, Y. P. Loh, S. Ong  
**Venue**: ISPACS 2022  
**DOI**: 10.1109/ISPACS57703.2022.10082804

**Method**: Extends zero-reference enhancement with noise awareness, specifically optimized for downstream object detection tasks.

**Key Contributions**:
- Noise-aware enhancement framework
- Task-specific optimization (object detection)
- Improved performance on downstream tasks

**Advantages**:
- Addresses noise issues
- Task-oriented design
- Practical application focus

**Disadvantages**:
- Task-specific (may not generalize to other applications)
- Less focus on aesthetic quality
- Limited handling of extreme scenarios

**Comparison with Our Approach**: This work shares our concern about noise, but our approach is more general-purpose, focusing on perceptual quality and handling multiple failure modes rather than optimizing for a specific downstream task.

---

#### 3.1.5 FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information

**Authors**: Recent work (2023)  
**ArXiv**: 2308.03033

**Method**: Utilizes Fourier frequency information to improve lightness and detail in low-light images. The approach estimates amplitude transform maps in the Fourier space and integrates global frequency with local spatial information.

**Key Contributions**:
- Frequency domain processing for low-light enhancement
- Amplitude transformation in Fourier space
- Integration of global frequency and local spatial information

**Advantages**:
- Novel frequency domain approach
- Effective lightness improvement
- Good detail preservation

**Disadvantages**:
- More complex processing pipeline
- May require FFT computations (computational overhead)
- Less focus on overexposure handling

**Comparison with Our Approach**: FourLLIE's frequency domain approach is interesting but different from our curve-based approach. Our texture-aware lighting maps could potentially benefit from frequency domain insights, but we maintain the simpler curve estimation framework for efficiency.

---

#### 3.1.6 ALEN: A Dual-Approach for Uniform and Non-Uniform Low-Light Image Enhancement

**Authors**: Recent work (2024)  
**ArXiv**: 2407.19708

**Method**: Introduces a classification mechanism to determine whether local or global illumination enhancement is required, followed by estimator networks for precise illumination and color adjustments.

**Key Contributions**:
- Adaptive classification of illumination conditions
- Dual approach for uniform and non-uniform lighting
- Improved adaptability to diverse scenarios

**Advantages**:
- Explicit handling of uniform vs. non-uniform lighting
- Adaptive enhancement strategy
- Good generalization

**Disadvantages**:
- Requires classification network (additional complexity)
- May not be zero-reference
- More parameters to train

**Comparison with Our Approach**: ALEN's adaptive approach is similar to our texture-aware and balanced enhancement concepts, but we maintain a simpler architecture without explicit classification. Our approach achieves adaptability through loss function design rather than architectural changes.

---

#### 3.1.7 Global Structure-Aware Diffusion Process for Low-Light Image Enhancement

**Authors**: Recent work (2023)  
**ArXiv**: 2310.17577

**Method**: Employs a diffusion-based framework with global structure-aware regularization to preserve details and enhance contrast, effectively suppressing noise and artifacts.

**Key Contributions**:
- Diffusion-based enhancement framework
- Global structure-aware regularization
- Effective noise and artifact suppression

**Advantages**:
- State-of-the-art enhancement quality
- Good detail preservation
- Effective noise suppression

**Disadvantages**:
- Diffusion models are computationally expensive
- Slower inference time
- May require paired or unpaired training data

**Comparison with Our Approach**: While diffusion models offer excellent quality, they sacrifice the speed and efficiency that make Zero-DCE practical. Our approach maintains efficiency while incorporating some structure-aware concepts through texture-aware lighting maps.

---

### 3.2 Comparative Analysis

| Method | Zero-Reference | Handles Extreme Low-Light | Handles Uneven Illumination | Handles Overexposure | Computational Efficiency | Aesthetic Focus |
|--------|---------------|---------------------------|----------------------------|---------------------|---------------------------|----------------|
| **Zero-DCE (Baseline)** | ✓ | ✗ | ✗ | ✗ | ✓✓✓ | ✗ |
| **Zero-LEINR** | ✓ | ✗ | ✗ | ✗ | ✓✓ | ✗ |
| **BEENet** | ✗ | ✗ | ✓ | ✓ | ✓✓ | ✓ |
| **Residual Quotient Learning** | ✓ | ? | ? | ? | ✓✓ | ✓ |
| **FourLLIE** | ? | ✓ | ? | ✗ | ✓ | ✓ |
| **ALEN** | ? | ✓ | ✓ | ? | ✓✓ | ✓ |
| **Diffusion-Based** | ✗ | ✓ | ✓ | ✓ | ✗ | ✓✓ |
| **Our Proposed Approach** | ✓ | ✓ | ✓ | ✓ | ✓✓ | ✓ |

**Legend**: ✓ = Yes, ✗ = No, ? = Not clearly stated, ✓✓✓ = Very efficient, ✓✓ = Efficient, ✓ = Moderate, ✗ = Slow

### 3.3 Advantages of Our Approach

1. **Comprehensive Problem Coverage**: Unlike most existing methods that address one or two issues, our approach simultaneously targets extreme low-light, uneven illumination, and overexposure.

2. **Maintains Zero-Reference Training**: We preserve the key advantage of Zero-DCE (no paired/unpaired data required) while addressing its limitations.

3. **Computational Efficiency**: Our improvements are primarily loss-based and fusion-based, maintaining the lightweight and fast inference properties of Zero-DCE.

4. **Perceptual Quality Focus**: Explicit optimization for aesthetic quality (NIMA) and naturalness (NIQE) distinguishes our approach from purely quantitative methods.

5. **Modular Design**: Each component (exposure fusion, texture-aware maps, balance loss, perceptual training) can be evaluated independently, allowing for incremental improvements.

6. **Practical Applicability**: Maintains real-time processing capability, making it suitable for practical applications unlike computationally expensive methods (e.g., diffusion models).

### 3.4 Potential Limitations and Mitigation

1. **Complexity of Fusion**: Exposure fusion may introduce computational overhead. *Mitigation*: Use efficient multi-scale fusion algorithms and optimize implementation.

2. **Loss Function Balancing**: Multiple loss terms may be difficult to balance. *Mitigation*: Use adaptive weighting schemes and extensive hyperparameter tuning.

3. **Training Stability**: Adding multiple loss components may affect training stability. *Mitigation*: Gradual integration of losses and careful learning rate scheduling.

4. **Generalization**: Improvements may be dataset-specific. *Mitigation*: Extensive testing on diverse datasets and cross-validation.

---

## 4. References

### 4.1 Baseline Paper

[1] C. Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 1777-1786, doi: 10.1109/CVPR42600.2020.00185

**Paper Link**: http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf

**GitHub Link**: https://github.com/Li-Chongyi/Zero-DCE

### 4.2 Surveyed Papers

[2] K. Ang, W. T. Lim, Y. P. Loh and S. Ong, "Noise-aware Zero-Reference Low-light Image Enhancement for Object Detection," 2022 International Symposium on Intelligent Signal Processing and Communication Systems (ISPACS), Penang, Malaysia, 2022, pp. 1-4, doi: 10.1109/ISPACS57703.2022.10082804

[3] W. H. Tang, H. Yuan, T. -H. Chiang and C. -C. Huang, "Zero-LEINR: Zero-Reference Low-light Image Enhancement with Intrinsic Noise Reduction," 2023 IEEE International Symposium on Circuits and Systems (ISCAS), Monterey, CA, USA, 2023, pp. 1-5, doi: 10.1109/ISCAS46773.2023.10181743

[4] J. Zhang, Y. Zhou, E. Zhu and J. Sun, "BEENet: A Balanced Enhancement Network for Uneven Exposure Low-Light Image Enhancement," 2024 6th International Conference on Frontier Technologies of Information and Computer (ICFTIC), Qingdao, China, 2024, pp. 647-650, doi: 10.1109/ICFTIC64248.2024.10913334

[5] C. Xie et al., "Residual Quotient Learning for Zero-Reference Low-Light Image Enhancement," in IEEE Transactions on Image Processing, vol. 34, pp. 365-378, 2025, doi: 10.1109/TIP.2024.3519997

[6] FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency Information. ArXiv: 2308.03033, 2023.  
**Link**: https://arxiv.org/abs/2308.03033

[7] ALEN: A Dual-Approach for Uniform and Non-Uniform Low-Light Image Enhancement. ArXiv: 2407.19708, 2024.  
**Link**: https://arxiv.org/abs/2407.19708

[8] Global Structure-Aware Diffusion Process for Low-Light Image Enhancement. ArXiv: 2310.17577, 2023.  
**Link**: https://arxiv.org/abs/2310.17577

[9] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement. ArXiv: 2409.01641, 2024.  
**Link**: https://arxiv.org/abs/2409.01641

### 4.3 Additional Resources

- Zero-DCE Project Page: https://li-chongyi.github.io/Proj_Zero-DCE.html
- PyIQA Library: https://github.com/chaofengc/IQA-PyTorch (for NIMA/NIQE metrics)
- Evaluation Metrics Documentation: Various image quality assessment papers and implementations

---

## 5. Implementation Timeline

### Phase 1: Survey and Analysis (Current - Week 1)
- ✅ Complete literature survey
- ✅ Identify key limitations and improvement opportunities
- ✅ Document findings and comparative analysis

### Phase 2: Baseline Evaluation (Week 2)
- Run comprehensive evaluation on test datasets
- Document quantitative and qualitative results
- Establish baseline metrics for comparison

### Phase 3: Implementation (Weeks 3-5)
- Week 3: Implement bright/dark balance loss
- Week 4: Implement texture-aware lighting maps
- Week 5: Implement exposure fusion and perceptual co-training

### Phase 4: Evaluation and Comparison (Week 6)
- Evaluate improved model on test datasets
- Compare results with baseline and other methods
- Document improvements and limitations

### Phase 5: Documentation and Submission (Week 7)
- Finalize implementation documentation
- Prepare results and visualizations
- Complete project report

---

## 6. Conclusion

This survey has identified the key limitations of the Zero-DCE baseline model and explored various approaches to address them. Our proposed methodology combines multiple complementary techniques to comprehensively address extreme low-light scenarios, uneven illumination, and overexposure while maintaining the efficiency and zero-reference training advantages of the original model.

The surveyed works provide valuable insights, but our approach offers a unique combination of:
- Comprehensive problem coverage
- Maintained computational efficiency
- Explicit perceptual quality optimization
- Modular, incremental improvement strategy

We anticipate that implementing these improvements will result in significant quantitative and qualitative enhancements, making the model more robust and practical for real-world low-light image enhancement applications.

