# Integrated Gradients

## Overview

This is my review of Integrated Gradients, starting from concepts of image gradients to integrated ones. 

# Table of Contents

1. [Introduction to Gradients](#gradients)

2. [Image Gradients](#image-gradients)

3. [Gradient Perturbation](#gradient-perturbation)

4. [Integrated Gradients](#integrated-gradients)

5. [References](#references)

---

## Introduction to Gradients

A gradient represents the rate of change of a function with respect to its inputs. In neural networks, gradients are used during the training process to update the model's parameters in the direction that minimizes the loss function. This process, known as backpropagation, efficiently computes these gradients layer by layer, enabling the model to learn from data.

---

## Image Gradients

Image gradients are a cornerstone in computer vision and image processing. They measure the directional change in intensity or color in an image, effectively capturing edges and contours. Mathematically, the gradient of an image is a vector composed of the partial derivatives with respect to the x and y directions:

$$
\nabla I = \begin{bmatrix} \frac{\partial I}{\partial x} \\\\ \frac{\partial I}{\partial y} \end{bmatrix}
$$

These gradients are instrumental in various applications, including edge detection, texture analysis, and object recognition. Techniques like the Sobel and Prewitt operators are commonly used to compute image gradients by convolving the image with specific kernels.

---

## Gradient Perturbation

Gradient perturbation involves introducing small changes to the input data or model parameters to analyze the model's sensitivity and robustness. This technique is widely used in:

- **Adversarial Training**: Crafting inputs with slight perturbations that can mislead the model, helping in developing defenses against such attacks.
- **Differential Privacy**: Adding noise to gradients during training to protect sensitive information in the dataset.
- **Optimization**: Escaping saddle points in the loss landscape by perturbing gradients, aiding in finding better minima during training.

Understanding gradient perturbation is crucial for developing robust and secure machine learning models.

---

## Integrated Gradients

Integrated Gradients (IG) is an interpretability technique designed to attribute the predictions of deep neural networks to their input features. It addresses the limitations of traditional gradient-based methods by considering the path from a baseline input to the actual input, providing more stable and theoretically grounded attributions.

### Conceptual Overview

- **Baseline Input**: IG introduces the concept of a baseline input, which represents an absence of features. Common choices include a black image (all zeros) for image models or an empty sequence for text models. The baseline serves as a reference point to measure the contribution of each feature.

- **Path Integration**: Instead of evaluating gradients at a single point, IG computes the integral of the gradients along a straight-line path from the baseline to the actual input. This approach captures how the model's output changes as each feature is gradually introduced, leading to more accurate attributions.

- **Axiomatic Properties**: IG satisfies several desirable axioms for interpretability methods:
  - *Sensitivity*: Features that significantly influence the output receive non-zero attributions.
  - *Implementation Invariance*: Functionally equivalent models yield identical attributions.
  - *Completeness*: The sum of attributions equals the difference between the model's output for the input and the baseline:

<div align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{IG}_i(x)&space;=&space;(x_i&space;-&space;x'_i)&space;\times&space;\int_{\alpha=0}^1&space;\frac{\partial&space;F(x'&space;&plus;&space;\alpha&space;\cdot&space;(x&space;-&space;x'))}{\partial&space;x_i}&space;d\alpha" alt="Integrated Gradients Formula"/>
</div>

This expression calculates the contribution of each input feature $x_i$ by averaging gradients taken along a path between the baseline $x'$ and the actual input $x$.

### Practical Implications

By providing clear insights into which features contribute to a model's prediction, IG enhances transparency and trust in AI systems. It's particularly useful in domains where understanding model decisions is critical, such as healthcare and finance.

---

## 5. References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- TensorFlow Integrated Gradients Tutorial: [tensorflow.org/tutorials/interpretability/integrated_gradients](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)
- Keras Integrated Gradients Example: [keras.io/examples/vision/integrated_gradients](https://keras.io/examples/vision/integrated_gradients/)
- Wikipedia - Image Gradient: [en.wikipedia.org/wiki/Image_gradient](https://en.wikipedia.org/wiki/Image_gradient)
- Simple Science on Gradient Perturbation: [simplescience.ai/en/gradient-perturbation--klkyll](https://simplescience.ai/en/gradient-perturbation--klkyll)

---