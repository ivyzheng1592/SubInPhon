# Generalization Notes

## Issue

The generalization split contains three-syllable words, while the training split contains only two-syllable words. In trial runs, the seq2seq decoder often outputs two-syllable strings for three-syllable generalization inputs.

This is likely a length-generalization problem rather than only a phonological-generalization problem. During training, the decoder repeatedly sees `<EOS>` after the two-syllable output shape, so it can learn a strong stopping prior:

```text
<SOS> syllable1 syllable2 <EOS>
```

At generalization time, the target sequence is longer, but the decoder can still predict `<EOS>` after two syllables. The recorder then converts predictions with `Alphabet.vec2word`, which stops reading at `<EOS>`, so the logged prediction appears as a two-syllable output.

The first intervention implemented in the code is EOS-loss downweighting during training. This is controlled by `hp.eos_loss_weight` or `--eos-loss-weight`. It is the least supervised option because it does not provide test-time length, template, or syllable-count information.

## Next Intervention: EOS Penalty Based On Syllable Nuclei

If EOS-loss downweighting does not solve the issue, add an inference-time EOS penalty that discourages stopping before the output contains enough syllable nuclei.

The core idea is:

```python
if predicted_vowel_count < source_vowel_count:
    output[:, eos_idx] -= eos_penalty
```

where:

- `source_vowel_count` is the number of vowel nuclei in the input UR.
- `predicted_vowel_count` is the number of vowel nuclei already emitted by the decoder.
- `eos_penalty` is a tunable positive value, such as `0.5`, `1.0`, or `2.0`.

This can be implemented as a soft penalty rather than a hard ban. A soft penalty still allows the model to emit `<EOS>` if it is very confident, but makes early stopping less attractive.

Why it helps:

- It directly targets the observed failure mode: early `<EOS>` after the two-syllable training shape.
- It is weaker than template conditioning because it does not tell the model the full output template, consonant positions, or vowel qualities.
- It is linguistically interpretable as a syllable-preservation bias: the process under study is vowel harmony, not syllable deletion or truncation.

Suggested evaluation contrast:

```text
baseline
EOS-loss downweighting
EOS-loss downweighting + nucleus-count EOS penalty
```

Useful diagnostics:

- generalization accuracy
- predicted syllable count
- predicted vowel/nucleus count
- parseability of predicted outputs
- early `<EOS>` rate
- overlong-output rate

## Later Intervention: Coverage-Based Decoding

If the nucleus-count EOS penalty is not enough, add coverage-based decoding. This is a larger code change because the decoder must track accumulated attention over source positions during prediction.

The core idea is to discourage `<EOS>` while substantial parts of the source remain under-attended:

```python
coverage = coverage + attention_t
undercoverage = compute_undercoverage(coverage)
output[:, eos_idx] -= coverage_penalty * undercoverage
```

Possible undercoverage definitions:

```python
undercoverage = torch.clamp(coverage_threshold - coverage, min=0).sum(dim=1)
```

or a simpler source-level criterion:

```python
undercoverage = (coverage < coverage_threshold).float().mean(dim=1)
```

Why it helps:

- The model is discouraged from terminating before it has attended across the source.
- This frames the problem as under-translation: the decoder stops after covering only the part of the UR corresponding to the two-syllable training distribution.
- It uses model-internal attention behavior rather than an explicit output template.
- It is common in attention-based seq2seq work for reducing under-translation, over-translation, and repetition.

References to revisit:

- Tu et al. 2016, "Modeling Coverage for Neural Machine Translation"
- Wu et al. 2016, "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
- Mi et al. 2016, "Coverage Embedding Models for Neural Machine Translation"
- See, Liu, and Manning 2017, "Get To The Point: Summarization with Pointer-Generator Networks"

Suggested evaluation contrast:

```text
baseline
EOS-loss downweighting
EOS-loss downweighting + nucleus-count EOS penalty
EOS-loss downweighting + coverage-based EOS penalty
```

Interpretation:

- Unconstrained decoding tests whether the model jointly learns the phonological mapping and length extrapolation.
- Nucleus-count EOS penalty tests whether the phonological pattern generalizes when premature termination is discouraged by a syllable-preservation bias.
- Coverage-based decoding tests whether premature termination is reduced when the model is encouraged to process the full UR before stopping.
