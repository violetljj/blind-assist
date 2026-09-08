"""Focused loss semantics checks, with no training experiment."""
import torch
from support_dice import masked_soft_dice


def checks(device):
    # Only map[0,0] participates: p=(.5,.5), target=(1,0), Dice loss=.5.
    labels = torch.tensor([[[[1., 0., -1.]], [[0., 0., -1.]]]], device=device)
    logits = torch.zeros_like(labels, requires_grad=True)
    loss = masked_soft_dice(logits, labels)
    assert torch.isclose(loss, torch.tensor(.5, device=device), atol=1e-6)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] < 0 and logits.grad[0, 0, 0, 1] > 0
    assert logits.grad[0, 0, 0, 2] == 0 and (logits.grad[0, 1] == 0).all()
    changed = logits.detach().clone()
    changed[labels == -1] = 80
    changed[0, 1] = 80
    assert torch.equal(masked_soft_dice(changed, labels), loss.detach())
    # Eligibility is per map, not pooled across the whole batch or both heads.
    two_logits = torch.cat([logits.detach(), torch.full_like(logits, 2.)])
    two_labels = torch.cat([labels, labels])
    expected = (masked_soft_dice(two_logits[:1], labels) +
                masked_soft_dice(two_logits[1:], labels)) / 2
    assert torch.allclose(masked_soft_dice(two_logits, two_labels), expected)
    for fill in [-1., 0.]:
        empty = torch.zeros_like(logits, requires_grad=True)
        value = masked_soft_dice(empty, torch.full_like(labels, fill))
        value.backward()
        assert value == 0 and (empty.grad == 0).all()
    target = torch.tensor([[[[1., 0.]]]], device=device)
    perfect = torch.tensor([[[[20., -20.]]]], device=device)
    spill = torch.tensor([[[[20., 20.]]]], device=device)
    missing = -perfect
    assert masked_soft_dice(perfect, target) < masked_soft_dice(spill, target)
    assert masked_soft_dice(spill, target) < masked_soft_dice(missing, target)
    print(f'PASS {device}: formula, positive eligibility, known-negative gradient, '
          'UNKNOWN zero gradient, per-map mean, empty maps, spill/miss ordering')


if __name__ == '__main__':
    checks('cpu')
    if torch.cuda.is_available():
        checks('cuda')
