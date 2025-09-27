Facial Expression Recognition & Affective Computing

Author: Masroor Rehan
Course: DS-D (I21-1707)
Project: Deep Learning Assignment

1. Overview

This project implements a multi-task deep learning model for facial expression recognition and affective computing. The model performs two tasks simultaneously:

Expression Classification: Predicts one of 8 facial expression categories.

Valence–Arousal Regression: Predicts continuous valence and arousal values (range [-1, 1]).

Two pretrained architectures were evaluated: ResNet50 and EfficientNetB0, with a comparison of their performance on a dataset of 3,999 annotated images.

2. Dataset

Total Samples: 3,999 images

Annotations per image:

Expression label → 8 classes (exp.npy)

Valence → Continuous value [-1,1] (val.npy)

Arousal → Continuous value [-1,1] (aro.npy)

Landmarks → Not used (lnd.npy)

Data Split:

Training: 3,399 images (85%)

Validation: 600 images (15%)

Stratified by expression labels to maintain class balance

3. Methodology
3.1 Preprocessing

Resize images to 224×224

Normalize to [0,1]

Data augmentations: flip, brightness, contrast

Labels:

Expression → one-hot encoded (8 classes)

Valence–Arousal → 2D vector [valence, arousal]

3.2 Models

(a) ResNet50

Pretrained on ImageNet

Global Average Pooling + Dropout

Two heads:

Classification: Dense(8, softmax)

Regression: Dense(2, tanh)

(b) EfficientNetB0

Pretrained on ImageNet

Stage 1: Base frozen, only dense heads trained

Stage 2: Fine-tuned last 20 layers with small learning rate

3.3 Training Setup

Optimizer: Adam

Loss:

Classification → Categorical Cross-Entropy

Regression → Mean Squared Error

Loss weights: Classification 1.0, Regression 0.5

Batch size: 32

Epochs: up to 15

Callbacks: Early stopping, Model checkpoint

4. Results
4.1 Expression Classification
Model	Val Accuracy	Macro F1	Cohen’s Kappa
ResNet50	43.1%	0.43	0.35
EfficientNetB0 (frozen)	12.5%	0.12	~0
EfficientNetB0 (fine-tuned)	31.0%	0.31	0.22

Observation: ResNet50 achieved the best classification performance.

Confusion: Misclassification occurred mostly between similar emotions (e.g., classes 0 vs 7, 2 vs 5).

4.2 Valence–Arousal Regression

ResNet50:

RMSE: Valence 0.39, Arousal 0.36

CCC: Valence 0.49, Arousal 0.39

SAGR: Valence 0.75, Arousal 0.78

Pearson Corr: Valence 0.57, Arousal 0.43

EfficientNetB0 (fine-tuned):

MSE: 0.164

MAE: 0.339

R²: 0.083

Interpretation:

Both models captured VA trends but explained low variance (R² < 0.1).

ResNet50 outperformed EfficientNetB0 in regression metrics.

4.3 Training Curves

ResNet50 reached peak validation accuracy ~43%

EfficientNetB0 (frozen) performed near random (12.5%)

Fine-tuning EfficientNetB0 improved accuracy to ~31%

5. Discussion

ResNet50 transferred better to the dataset and achieved stronger performance.

EfficientNetB0 required fine-tuning and still underperformed.

Dataset size (~4k images) is relatively small for EfficientNetB0’s high capacity.

Emotion recognition is inherently ambiguous, causing class confusion.

Regression captures general trends but struggles with exact values.

6. Conclusion

Best Overall Model: ResNet50

Classification Accuracy: 43%

Stronger Valence–Arousal metrics

EfficientNetB0 improves with fine-tuning but remains less effective.

7. References

He, K. et al., "Deep Residual Learning for Image Recognition", CVPR 2016.

Tan, M. & Le, Q., "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.

Multi-task learning and affective computing literature.
