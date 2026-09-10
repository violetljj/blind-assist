"""Loss-side native positive pooling; predictor and negative objectives unchanged."""
import torch
from torch.nn import functional as F

from mz15_train import balanced_local


def loss_for(out, cell_truth, cell_known, frame_truth, frame_known):
    known = cell_known.bool() & out['candidate_mask'][None]
    truth = cell_truth.bool()
    local = balanced_local(out['field'], truth, known[..., None])
    positive_cells = truth & known[..., None]
    witness = positive_cells.flatten(1, 2).any(1)
    frame_positive = frame_truth.bool()
    mask = frame_known.bool() & known.flatten(1).any(1)[:, None] & (~frame_positive | witness)
    native = out['field'].masked_fill(~positive_cells, -torch.inf).flatten(1, 2).amax(1)
    # No-witness rows are excluded as before; keep every intermediate BCE finite.
    native = torch.where(witness, native, torch.zeros_like(native))
    raw = torch.where(frame_positive & witness, native, out['OPEN']['raw'])
    values = F.binary_cross_entropy_with_logits(raw, frame_truth.float(), reduction='none')
    query = (values * mask).sum() / mask.sum().clamp_min(1)
    return local + .25 * query, dict(local=local, query=query,
        skipped_positive=(frame_known.bool() & frame_positive & ~witness).sum(),
        native_query_active=(mask & frame_positive).sum())


def self_test():
    from mz54_full_rgb_model import loss_for as original
    from mz54_full_rgb_model import RasterQuery

    def output(field, candidate):
        return dict(field=field, candidate_mask=candidate,
                    OPEN=RasterQuery.pool(field, candidate))

    checks = []
    # Query 0 has one true witness and a stronger known-negative/UNKNOWN winner.
    for winner_known in (True, False):
        field = torch.zeros((1, 1, 3, 4), requires_grad=True)
        with torch.no_grad():
            field[0, 0, 1, 0] = 4
            field[0, 0, 2, 1:] = 2
        candidate = torch.ones((1, 3), dtype=torch.bool)
        cell_truth = torch.zeros_like(field, dtype=torch.bool); cell_truth[0, 0, 0, 0] = True
        cell_known = torch.tensor([[[True, winner_known, True]]])
        truth = torch.tensor([[True, False, False, False]])
        known = torch.ones_like(truth)
        out = output(field, candidate)
        _, old_parts = original(out, cell_truth, cell_known, truth, known)
        _, parts = loss_for(out, cell_truth, cell_known, truth, known)
        torch.testing.assert_close(parts['local'], old_parts['local'], atol=0, rtol=0)
        old_gradient = torch.autograd.grad(old_parts['query'], field, retain_graph=True)[0]
        gradient = torch.autograd.grad(parts['query'], field, retain_graph=True)[0]
        assert old_gradient[0, 0, 1, 0] < 0 and old_gradient[0, 0, 0, 0] == 0
        assert gradient[0, 0, 1, 0] == 0 and gradient[0, 0, 0, 0] < 0
        torch.testing.assert_close(gradient[..., 1:], old_gradient[..., 1:], atol=0, rtol=0)
        checks.append('Known-negative winner' if winner_known else 'UNKNOWN winner')
    # With only negative queries, the complete loss and gradient must be exact.
    torch.manual_seed(60)
    field = torch.randn(2, 2, 3, 4, requires_grad=True)
    candidate = torch.tensor([[True, False, True], [True, True, True]])
    cell_truth = torch.zeros_like(field, dtype=torch.bool)
    cell_known = torch.ones((2, 2, 3), dtype=torch.bool)
    truth = torch.zeros((2, 4), dtype=torch.bool); known = torch.ones_like(truth)
    out = output(field, candidate)
    old_loss, _ = original(out, cell_truth, cell_known, truth, known)
    loss, parts = loss_for(out, cell_truth, cell_known, truth, known)
    torch.testing.assert_close(loss, old_loss, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(loss, field, retain_graph=True)[0],
                               torch.autograd.grad(old_loss, field, retain_graph=True)[0], atol=0, rtol=0)
    checks.append('All-negative complete loss/gradient identity')
    # A positive outside the candidate field or with no known witness stays skipped.
    for all_unknown in (False, True):
        positive = torch.ones_like(truth)
        known_cells = torch.zeros_like(cell_known) if all_unknown else cell_known
        loss, parts = loss_for(out, cell_truth, known_cells, positive, known)
        old_loss, old_parts = original(out, cell_truth, known_cells, positive, known)
        assert torch.isfinite(loss) and parts['skipped_positive'] == 8
        torch.testing.assert_close(loss, old_loss, atol=0, rtol=0)
        gradient = torch.autograd.grad(loss, field, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
        if all_unknown:
            assert loss == 0 and torch.count_nonzero(gradient) == 0
    checks.append('No-witness/all-UNKNOWN finite skip identity')
    # Native truth in a masked candidate cannot receive query gradient.
    cell_truth[0, 0, 1, 0] = True
    _, parts = loss_for(out, cell_truth, cell_known, torch.ones_like(truth), known)
    assert parts['native_query_active'] == 0
    checks.append('Candidate-mask exclusion')
    print('PASS', '; '.join(checks))


if __name__ == '__main__':
    self_test()
