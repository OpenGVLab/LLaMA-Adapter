# LLaMA-Adapter-V2-65B chat model zoo

This page provides a summary of all released models. Download links are also included. We will continue to release improved models and new architectures.

## Bias-scale-norm tuning

**Description:** For the bias-scale-norm tuned models, a learnable channel-wise scaling factor is added before the fully connected layers, and a learnable channel-wise bias is added after the fully connected layers. The parameters in the normalization layers are also tuned.

| Version | Training data | Download url | Notes |
| --- | --- | --- | --- |
| v1 | ShareGPT | [Google drive](https://drive.google.com/file/d/1EGDVyXKNt2k9rApoXQY2i0Wm34OatkgK/view?usp=sharing) | Initial version |

## Bias-scale-norm tuning + LoRA

Coming soon