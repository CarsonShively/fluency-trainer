# Objective

The goal was to build a model that could accurately represent a user's actual speech compared to a target sentence and grade it according to three categories: accuracy, fluency, and prosodic, along with a final total score that is assessed independently rather than as a sum of the three.


# Modeling Requirements

Many speech models focus on inferring mispronounced words rather than assessing the quality of what was actually spoken. The goal was to accurately grade what was spoken by the user compared with the target they were attempting to say.

The model needed to capture several difficult patterns, including incorrect ordering, extra sounds, missing sounds, variations in speaker cadence when pronouncing the same word, and more.


# Earlier Architecture: Cross Attention

An earlier approach used cross attention where the target phones were the query and the user audio frames were the key and value.

The architecture was effective at determining if a phone was present but struggled to capture key patterns needed for this goal. The primary pattern this cross attention misses is sequential order; earlier phones can attend to later user audio frames.

While there are methods that could be used to overcome this, cross attention is not a natural fit for capturing the key patterns of this task. Opposed to trying to force an unnatural architecture to work, I decided to test another architecture that captures these patterns more natively and explicitly.

# Architecture

## 1. Alignment Matrix:
Target phone embeddings and user audio logits are projected into the same feature dimension. This allows for matrix multiplication, where each dot product is one cell representing the similarity score between each target phone and user audio frame.

The core idea is that correctly pronounced speech should produce a diagonal alignment pattern, while mispronunciation patterns such as missing, extra, out of order, or incorrect phones will result in deviations from this diagonal.

## 2. Residual Convolution Network
The convolutional layers capture local patterns around each target phone, allowing each phone to incorporate the context from its corresponding audio frames.

## 3. Audio Axis Compression
The audio axis is mean pooled, collapsing the matrix into a sequence of context-aware phone feature vectors.

## 4. Bidirectional LSTM
A bidirectional LSTM processes these vectors in both directions, allowing the model to capture relationships between neighboring phonemes.

## 5. Alignment Vector
The BLSTM output is mean pooled, collapsing the phone axis into a fully context-aware feature vector that summarizes the complete target-to-audio alignment.

## 6. Multi-Task MLP Regression Head
The alignment vector is then passed through three separate regression heads that predict accuracy, fluency, and prosodic scores. These scores are then passed into a final regression head that predicts the total score.




# Metrics
**Loss Metric:** Mean Squared Error

**Loss Calculation:** Accuracy Loss + Fluency Loss + Prosodic Loss + Total Loss

**Mean Baseline Benchmark:** 42.72442626953125

**Holdout Evaluation:** 5.25933837890625

**Model Lift:** 87.69%


# Run Demo
```bash
1. make install
2. make demo
```