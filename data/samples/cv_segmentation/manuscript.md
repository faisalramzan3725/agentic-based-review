# LiteSeg: Lightweight Attention for Efficient Skin Lesion Segmentation

## Abstract

Accurate segmentation of skin lesions in dermoscopic images supports early screening, but state-of-the-art segmentation networks are too heavy for mobile devices. We propose LiteSeg, an encoder-decoder network that inserts lightweight channel attention into skip connections and replaces standard convolutions with depthwise separable blocks. On the International Skin Imaging Collaboration (ISIC) 2018 benchmark, LiteSeg reaches a Dice similarity coefficient (DSC) of 0.912 while using 38% fewer parameters than the strongest baseline. Results are averaged over five random seeds and the improvement over U-Net is statistically significant under a paired t-test. An ablation study shows that the attention module and the separable blocks contribute complementary gains. LiteSeg runs at 41 frames per second on a mid-range phone, which makes on-device lesion screening practical. Code and trained weights are released to support reproducibility.

## 1. Introduction

Skin cancer is among the most common cancers worldwide, and dermoscopic imaging is widely used for screening. Automatic lesion segmentation is an important first step for downstream classification [1]. Convolutional encoder-decoder models such as U-Net [2] dominate this task, and transformer models have recently improved accuracy further [3]. However, these models require substantial memory and compute, which limits deployment on phones in low-resource settings.

In this paper we propose LiteSeg, a compact segmentation network designed for on-device inference. Our contributions are threefold. First, we introduce a lightweight channel attention module placed on skip connections. Second, we combine it with depthwise separable convolutions and an ASPP bottleneck to preserve receptive field at low cost. Third, we provide a rigorous evaluation with repeated runs, significance testing and an ablation study.

## 2. Related Work

Encoder-decoder networks with skip connections were popularised by U-Net [2] and later extended with residual blocks [4]. Attention mechanisms have been added to skip connections to suppress irrelevant background [5], and lightweight attention modules for medical image segmentation were explored in [6]. Vision transformers [3] and hybrid models [7] achieve strong accuracy but at high computational cost. Efficient architectures based on depthwise separable convolutions [8] reduce parameters for mobile vision. Our work differs from [6] by combining attention with separable blocks and by targeting phone-class hardware with measured latency.

## 3. Methods

LiteSeg follows an encoder-decoder design with four resolution levels (Figure 1). Each encoder stage uses two depthwise separable convolution blocks with batch normalisation and ReLU activation. Skip connections pass through a lightweight channel attention module that pools features globally, applies a one-dimensional convolution and rescales channels. The bottleneck uses an ASPP block with dilation rates 1, 6 and 12.

We use the ISIC 2018 dataset with 2594 training images, split into 2075 training and 519 validation images, and evaluate on the official 1000-image test set. Images are resized to 256 x 256 pixels and augmented with flips, rotations and colour jitter. Implementation details: we train with the AdamW optimizer, learning rate 0.001 with cosine decay, batch size 16, for 150 epochs on a single GPU. All hyperparameters were selected on the validation split. Every experiment is repeated with five random seeds.

We compare against U-Net [2], Attention U-Net [5], the lightweight attention network of [6] and a vision transformer baseline [3]. We report the Dice similarity coefficient and intersection over union as mean ± standard deviation over seeds, and test differences with a paired t-test.

## 4. Results

Table 1 summarises segmentation accuracy and model size on the ISIC 2018 test set. LiteSeg achieves a DSC of 0.912 ± 0.004, compared with 0.894 ± 0.006 for U-Net and 0.905 ± 0.005 for the lightweight attention network. The improvement over U-Net is significant (paired t-test, p < 0.01). LiteSeg uses 38% fewer parameters than the transformer baseline and requires 2.1 billion floating point operations per image.

| Model | DSC | Parameters (M) |
|---|---|---|
| U-Net | 0.894 | 7.8 |
| Attention U-Net | 0.901 | 8.7 |
| Transformer baseline | 0.910 | 4.2 |
| LiteSeg | 0.912 | 2.6 |

Table 2 reports the ablation study. Removing the attention module lowers DSC to 0.903, and replacing separable blocks with standard convolutions increases parameters by 2.9 times with no accuracy gain. On a mid-range phone LiteSeg runs at 41 frames per second. Figure 2 shows qualitative examples, including lesions with low contrast and hair artefacts.

| Variant | DSC |
|---|---|
| LiteSeg | 0.912 |
| without attention | 0.903 |
| without separable blocks | 0.911 |

## 5. Discussion

The results indicate that channel attention on skip connections recovers most of the accuracy lost by compressing the encoder. The ablation study confirms that attention and separable blocks contribute complementary effects. The per-seed variance is small, which suggests the gains are stable. A limitation of this study is that we evaluate on a single public benchmark; images from other devices and populations may differ in colour and resolution. We also did not evaluate robustness to adversarial perturbations, which future work should address.

## 6. Conclusion

We presented LiteSeg, a lightweight segmentation network for dermoscopic images. It achieves a DSC of 0.912 with 38% fewer parameters than the strongest baseline and runs in real time on a phone.

## Declarations

Data availability: The ISIC 2018 dataset is publicly available from the challenge website.
Code availability: Code is available at https://github.com/example/liteseg.
Conflict of interest: The authors declare no competing interests.

## References

[1] Codella N. et al. Skin lesion analysis toward melanoma detection. 2018.
[2] Ronneberger O. et al. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015.
[3] Dosovitskiy A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 2021.
[4] He K. et al. Deep Residual Learning for Image Recognition. 2016.
[5] Oktay O. et al. Attention U-Net: Learning where to look for the pancreas. 2018.
[6] Chen L. et al. Lightweight Attention Modules for Efficient Medical Image Segmentation. 2020.
[7] Chen J. et al. TransUNet: Transformers make strong encoders for medical image segmentation. 2021.
[8] Howard A. et al. MobileNets: Efficient convolutional neural networks for mobile vision applications. 2017.
[9] Tschandl P. et al. The HAM10000 dataset of dermatoscopic images. 2018.
[10] Isensee F. et al. nnU-Net: a self-configuring method for biomedical image segmentation. 2021.
[11] Xie E. et al. SegFormer: Simple and efficient design for semantic segmentation. 2021.
[12] Liu Z. et al. A ConvNet for the 2020s. 2022.
[13] Kirillov A. et al. Segment Anything. 2023.
