"""Focused first-peak fixed-decoder checks; no training or scientific jobs."""
import unittest

import numpy as np
import torch

from cnh_bin_summary_control_dev import decode_summary


class BinSummaryTests(unittest.TestCase):
    def test_first_peak_not_largest_and_total_preserved(self):
        z = torch.zeros(1, 8, 8, 8, 16)
        z[:, -1, :, :, 3], z[:, -1, :, :, 9] = 4., 12.
        decoded = decode_summary(z, [1])
        assert decoded[0, -1, 0, 0, 3] == 4.
        torch.testing.assert_close(decoded[0, -1, 0, 0, 9], torch.tensor(12./15))
        torch.testing.assert_close(decoded.sum(-1), z.sum(-1), rtol=1e-6, atol=2e-6)
        assert torch.count_nonzero(decoded[:, :-1]) == 0

    def test_plateau_leftmost_and_endpoint(self):
        z = torch.zeros(1, 8, 8, 8, 16)
        z[:, -1, :, :, 4:7] = 5.
        decoded = decode_summary(z, [1])
        assert decoded[0, -1, 0, 0, 4] == 5.
        torch.testing.assert_close(decoded[0, -1, 0, 0, 5], torch.tensor(10./15))
        for endpoint in (0, 15):
            z.zero_()
            z[:, -1, :, :, endpoint] = 3.
            decoded = decode_summary(z, [1])
            assert decoded[0, -1, 0, 0, endpoint] == 3.
            assert decoded[0, -1, 0, 0].sum() == 3.

    def test_missing_peak_signed_total_and_nan_padding(self):
        torch.manual_seed(1)
        z = torch.randn(2, 8, 8, 8, 16).clamp_max(2.99)
        lengths = torch.tensor([2, 5])
        expected = decode_summary(z, lengths)
        for i, length in enumerate(lengths):
            torch.testing.assert_close(expected[i, -1],
                (z[i, -1].sum(-1)/16)[..., None].expand(8, 8, 16))
            z[i, :8-length] = float('nan')
        actual = decode_summary(z, lengths)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()

    def test_independent_numpy_decoder(self):
        torch.manual_seed(2)
        z = torch.randn(1, 8, 8, 8, 16)*4
        actual = decode_summary(z, [8]).numpy()
        data = z.numpy()
        expected = np.empty_like(data)
        for t in range(8):
            for y in range(8):
                for x in range(8):
                    bins = data[0, t, y, x]
                    peaks = [r for r in range(16) if bins[r] >= 3
                        and bins[r] > (bins[r-1] if r else -np.inf)
                        and bins[r] >= (bins[r+1] if r < 15 else -np.inf)]
                    if peaks:
                        first = peaks[0]
                        expected[0, t, y, x] = (bins.sum()-bins[first])/15
                        expected[0, t, y, x, first] = bins[first]
                    else:
                        expected[0, t, y, x] = bins.sum()/16
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_cpu(self):
        torch.manual_seed(3)
        z = (torch.randn(2, 8, 8, 8, 16)*4).half()
        lengths = torch.tensor([1, 8])
        torch.testing.assert_close(decode_summary(z.cuda(), lengths.cuda()).cpu(),
                                   decode_summary(z, lengths), rtol=1e-5, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
