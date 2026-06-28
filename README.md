# Causal Transformer Neural Decoder Demo

This is a runnable architectural prototype for replacing an RNN-based neural
decoder with a lightweight causal Transformer. It models the same general task
described in the electrode-pooling work:

```text
binned neural activity [trial, time, channel]
    -> forelimb position and velocity [trial, time, output]
```

The word **decoder** here means a brain-computer-interface model that decodes
kinematics from neural activity. The implementation uses a Transformer encoder
stack with a causal attention mask; it does not use the text-generation-style
`TransformerDecoder` module because there is no separate target-token stream.

## Quick verification without research data

Python 3.10+ and PyTorch 2.x are recommended.

```powershell
python decoder_demo.py --quick
python -m unittest -v test_decoder_demo.py
```

The first command generates a small synthetic neural dataset, trains both the
Transformer and GRU baseline, and writes checkpoints plus `results.json` to
`artifacts/`. The test suite also verifies that changing future neural samples
cannot alter predictions at earlier timesteps.

## Run a longer synthetic comparison

```powershell
python decoder_demo.py --model both --epochs 10
```

The synthetic scores only prove that the code and optimization pipeline work.
They are not evidence that the Transformer is better for the real experiment.

## Use real data later

Prepare a NumPy `.npz` file:

```python
import numpy as np

np.savez(
    "animal_01.npz",
    neural=neural_array,          # float32 [trials, time_bins, channels]
    kinematics=kinematic_array,  # float32 [trials, time_bins, 4 or 6]
)
```

Then run:

```powershell
python decoder_demo.py --data animal_01.npz --model both --epochs 50
```

For `[x, y, vx, vy]`, the last dimension is 4. For
`[x, y, z, vx, vy, vz]`, it is 6. Train/validation/test splitting is performed
at the trial level, and normalization statistics are calculated from training
trials only.

`--pool-size N` sums fixed groups of N feature channels. This is a convenient
software proxy for testing dimensions, not a faithful replacement for the
paper's raw-signal electrode-pooling pipeline. If the research team already has
pooled spike features, pass those directly and leave `--pool-size 1`.

## Fair RNN-versus-Transformer experiment

Use identical preprocessing, trial splits, temporal bin size, history window,
and output definitions. Compare at least:

- per-output and mean R-squared;
- RMSE in physical units after inverse normalization;
- variability across repeated seeds or cross-validation folds;
- parameter count, inference latency, and causal/online behavior;
- performance for individual-electrode, pooled-electrode, and channel-matched
  discard conditions.

Because the available dataset may be small and animal-specific, a Transformer
is not guaranteed to outperform the RNN. The useful research question is
whether attention improves long-range temporal modeling or robustness under
electrode pooling while retaining acceptable latency and sample efficiency.
