# Generalization Notes

## Issue

The generalization split contains three-syllable words, while the training split contains only two-syllable words. In trial runs, the seq2seq model does not reliably generalize from two-syllable training words to three-syllable generalization words.

The initial symptom looked like premature stopping: generalization outputs often appeared to contain only two syllables. This suggested an EOS bias, because the model repeatedly sees `<EOS>` after two syllables during training.

However, later diagnostics suggest that EOS is not the main cause.

## Updated Diagnosis

EOS-loss downweighting was tested, including the extreme setting:

```text
eos_loss_weight = 0.0
```

This removes the training loss pressure to predict EOS. The result was informative:

- For two-syllable words, the beginning of the generated output is correct when manually matched to the input length.
- For three-syllable words, the output starts to become wrong around the third syllable.
- Attention plots show nearly one-to-one alignment for the first two syllables.
- Starting at the third syllable, attention becomes messy and diffuse.

This suggests that the core problem is not early EOS generation. The model can continue decoding, but it does not maintain a clean alignment procedure beyond the two-syllable length regime seen in training.

In other words, the failure is better described as length extrapolation of attention/alignment:

```text
training teaches:
  align and transduce two syllables

generalization requires:
  keep applying the same alignment/transduction routine to a third syllable
```

The model appears to learn a bounded two-syllable procedure rather than an iterable left-to-right phonological mapping.

## Deprecated Interventions

### EOS Nucleus Penalty

Do not prioritize EOS nucleus penalty for now.

The idea was to penalize EOS until the decoded output contained enough vowel nuclei:

```python
if predicted_vowel_count < source_vowel_count:
    output[:, eos_idx] -= eos_penalty
```

This is now less promising because the model's failure is not simply that it stops too early. If alignment has already degraded by the third syllable, forcing or encouraging the decoder to continue will likely produce longer but still incorrect outputs.

### Plain Coverage-Based Decoding

Plain coverage-based decoding is also less promising than originally thought.

Coverage helps when the model skips source material or repeatedly attends to the same positions. But the current attention plots suggest a different problem: the model does attend to the third-syllable region, yet attention there is diffuse rather than sharply aligned.

So the issue is not:

```text
the model ignores the third syllable
```

It is closer to:

```text
the model reaches the third syllable but cannot preserve precise local alignment there
```

Coverage may still be useful later, but it is not the most targeted next intervention.

## Updated Recommendations

The next interventions should target attention structure directly. The goal is to preserve a stable, mostly monotonic, local alignment beyond the two-syllable training length.

### 1. Diagonal Attention Regularization

This is the preferred next intervention.

Keep the current attention mechanism, but add an auxiliary loss that rewards attention mass near a diagonal alignment. The model is still free to learn the output forms, but it is softly encouraged to keep input-output correspondence local and left-to-right.

Why it helps:

- Directly targets the observed failure: diffuse attention at the third syllable.
- Does not provide an output template.
- Does not provide the correct output length at decoding time.
- Is linguistically interpretable as a bias toward local, order-preserving phonological correspondence.
- Is less invasive than replacing the attention mechanism.

A soft Gaussian target is preferable to a hard one-hot diagonal. This allows local flexibility while discouraging highly diffuse attention.

Conceptually:

```text
small penalty:
  attention is sharp and near the expected source position

large penalty:
  attention is diffuse or far from the expected source position
```

### 2. Location-Aware Attention

Location-aware attention changes the attention mechanism so that the current attention decision depends partly on previous attention.

Normal attention asks:

```text
given the decoder state, where should I attend?
```

Location-aware attention asks:

```text
given the decoder state and where I attended before, where should I attend now?
```

Why it helps:

- Encourages the model to learn a smooth alignment trajectory.
- May help preserve the successful first-two-syllable alignment strategy into later syllables.
- Adds an architectural bias without explicitly specifying a template or target length.

Downside:

- More invasive than diagonal regularization.
- May still struggle if the training distribution never requires longer alignment trajectories.

### 3. Local / Windowed Attention

Local or windowed attention restricts, or softly biases, each decoder step to attend near an expected source position.

Why it helps:

- Prevents attention from becoming globally diffuse.
- Encodes the phonological assumption that mappings are mostly local and order-preserving.
- More targeted than coverage when the model already reaches the right region but loses sharpness.

Downside:

- Requires defining the expected alignment center.
- Could be too restrictive if the model needs flexibility for insertions, deletions, or non-identical source/target lengths.

### 4. Full Monotonic Attention

Full monotonic attention is the strongest structural intervention. It changes attention so that the alignment generally moves forward through the source and does not freely jump backward.

Why it helps:

- Closely matches the expected left-to-right nature of most phonological transductions.
- Directly attacks the failure to continue alignment beyond the training length.

Downside:

- Largest architectural change among the proposed options.
- More implementation and debugging work.
- May be unnecessarily strong for the current problem.

## Current Ranking

Recommended order:

```text
1. Diagonal attention regularization
2. Location-aware attention
3. Local/windowed attention
4. Full monotonic attention
```

The most natural next experiment is diagonal attention regularization, because it directly targets the messy third-syllable attention while preserving the model's autonomy over the generated phonological form.
