import unittest

import numpy as np
import torch

from decoder_demo import (
    CausalTransformerKinematicDecoder,
    generate_synthetic_data,
    pool_channels,
)


class DecoderDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = CausalTransformerKinematicDecoder(
            input_dim=8,
            output_dim=4,
            d_model=16,
            nhead=4,
            layers=1,
            feedforward_dim=32,
            dropout=0.0,
        ).eval()

    def test_forward_shape(self) -> None:
        neural = torch.randn(3, 12, 8)
        self.assertEqual(tuple(self.model(neural).shape), (3, 12, 4))

    def test_causal_mask_blocks_future_information(self) -> None:
        original = torch.randn(2, 10, 8)
        changed_future = original.clone()
        changed_future[:, 6:] += 100.0
        with torch.no_grad():
            first = self.model(original)
            second = self.model(changed_future)
        torch.testing.assert_close(first[:, :6], second[:, :6], atol=1e-5, rtol=1e-5)

    def test_synthetic_data_and_pooling(self) -> None:
        neural, kinematics = generate_synthetic_data(5, 12, 10, 4, seed=1)
        pooled = pool_channels(neural, pool_size=3)
        self.assertEqual(neural.shape, (5, 12, 10))
        self.assertEqual(kinematics.shape, (5, 12, 4))
        self.assertEqual(pooled.shape, (5, 12, 4))
        self.assertTrue(np.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
