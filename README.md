# Automatic-Bowel-Sound-Detection-and-Classification-using-Wav2Vec2
# Introduction

Bowel sound analysis is a valuable tool in gastrointestinal health monitoring. Traditional approaches rely on manual auscultation, which is time-consuming and subjective. This project proposes an automatic system for detecting and classifying bowel sounds from audio recordings. The system is designed to identify the presence of bowel sounds at a fine-grained, frame-level resolution and to classify the detected sounds into meaningful categories, facilitating objective analysis and potential clinical applications.

# Overview

The proposed approach leverages a pretrained Wav2Vec2 model as a feature extractor. The system performs two main tasks:

Frame-level detection: Determine whether a bowel sound is present in short audio frames (bowel sound vs. no bowel sound).

Segment-level classification: Classify detected bowel sound segments into one of three categories:

b → Single Burst

mb → Multiple Burst

h → Harmonic

This multi-task approach enables the model to both localize bowel sounds accurately and classify them reliably.

# Methodology
- Data Preparation

Audio recordings are resampled to 16 kHz.

Each recording is split into 5-second chunks to facilitate batch processing.

Chunks are further divided into 20 ms frames.

Ground-truth annotations are converted into frame-level labels and segment-level labels for classification.

- Model Architecture

Backbone: Pretrained Wav2Vec 2.0

A transformer-based model pretrained on raw audio.

Extracts deep, high-level audio features from the waveform.

Output: Hidden representations per frame with dimensions [batch_size, time_steps, hidden_size].

Purpose: Capture temporal patterns and acoustic characteristics crucial for bowel sound detection.


Frame-Level Head

Simple Linear layer on top of the backbone’s hidden states.

Input: Hidden state of each frame.

Output: Single logit per frame indicating presence (1) or absence (0) of a bowel sound.

This head allows the model to detect exactly where in time a sound occurs.

Segment-Level Head

Another Linear layer operating on mean-pooled hidden states of detected segments.

Input: Average hidden state across frames in a segment.

Output: Three logits per segment, corresponding to classes:

b: single burst

mb: multiple bursts

h: harmonic sound

Purpose: Classify the type of bowel sound after detecting its occurrence.

Overall Workflow

Audio → Wav2Vec2 → frame-level logits for detection → segment-level logits for classification.

Loss combines frame-level (BCE + Soft IoU) and segment-level (weighted Cross-Entropy) for end-to-end training.

- Loss Functions

A multi-task loss is used to optimize both detection and classification tasks:

Frame-Level Loss :

BCEWithLogitsLoss: Treats each small frame as sound (1) or no sound (0). Encourages the model to predict the presence of bowel sounds accurately.

Soft IoU Loss: Measures how well predicted frames overlap with true sound frames. Helps improve detection, especially when the dataset is imbalanced.

Segment-Level Loss:

Weighted Cross-Entropy: Classifies detected sound segments into types (b, mb, h). Class weights account for imbalance between segment types, ensuring rarer classes are learned properly.

Total Loss:

Simply sums frame-level and segment-level losses. This allows the model to learn detection and classification together, improving overall performance.

- Training Strategy

Optimizer: Adam with learning rate 1e-5

Weight decay: 1e-4 (L2 regularization to reduce overfitting)

Gradient clipping: 1.0

Data split: 80% training, 20% validation

# Results
- Frame-level Metrics (Sound vs No Sound)

F1 Score: 0.777

Precision: 0.684

Recall: 0.898

Accuracy: 0.794

The high recall indicates that the system reliably detects most bowel sounds, minimizing missed events.

- Segment-level Metrics (Sound Type Classification)

Accuracy: 0.664

F1 Score (macro): 0.607

Precision (macro): 0.629

Recall (macro): 0.593

While the system performs well in detecting sounds, distinguishing between multiple burst and harmonic sounds remains more challenging.

- Loss and Accuracy Curves


<img width="1000" height="500" alt="Figure_13" src="https://github.com/user-attachments/assets/31d31b92-c157-4873-8abd-c2d15da1563d" />

<img width="1000" height="500" alt="Figure_14" src="https://github.com/user-attachments/assets/4fb542be-e3d0-4d84-bea4-d0177a385ac6" />

The model demonstrates steady convergence on the training data, with the training loss decreasing from ~1.8 to ~0.8 over 20 epochs. The validation loss decreases initially but plateaus around ~1.0, indicating the onset of slight overfitting.

At the frame level, the model achieves high accuracy (~91%), showing strong capability in detecting the presence of bowel sounds. Segment-level classification (b, mb, h) is more challenging, reaching a maximum accuracy of ~82%, reflecting the increased complexity and possible class imbalance in the dataset.

Overall, the results suggest that while the model effectively identifies bowel sound events, further improvements in segment-level classification could be achieved through data augmentation.

# Conclusion

The proposed system demonstrates that a pretrained Wav2Vec2 backbone, combined with lightweight detection and classification heads, can effectively detect and classify bowel sounds from audio recordings. Frame-level detection achieves high recall, ensuring that most bowel sounds are identified, while segment-level classification provides meaningful categorization of detected sounds.

# Future Developments

Potential improvements include:

Expanding the dataset to improve segment-level classification.

Implementing attention-based segment classifiers for better discrimination between similar sound types.

Exploring data augmentation techniques to increase variability and robustness.

Integrating the model into a real-time bowel sound monitoring system for clinical or research applications.
