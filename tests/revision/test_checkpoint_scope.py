import pytest

torch = pytest.importorskip("torch")

from clinhallu.engine.checkpoints import (  # noqa: E402
    STATE_SCOPE_TRAINABLE,
    find_latest_checkpoint,
    load_model_state,
    save_checkpoint,
    trainable_state_dict,
)


class Net(torch.nn.Module):
    """Mimics the real layout: frozen ``encoder.*`` base + LoRA + head."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = torch.nn.ModuleDict(
            {"base": torch.nn.Linear(8, 8), "lora_A": torch.nn.Linear(8, 2)}
        )
        for parameter in self.encoder["base"].parameters():
            parameter.requires_grad = False
        self.head = torch.nn.Linear(8, 1)


def test_trainable_state_drops_only_the_frozen_backbone():
    state = trainable_state_dict(Net())
    assert not any(k.startswith("encoder.base") for k in state)
    assert {"encoder.lora_A.weight", "head.weight", "head.bias"} <= set(state)


def test_round_trip_restores_trainable_weights_and_keeps_rebuilt_backbone():
    trained, fresh = Net(seed=1), Net(seed=1)  # same pinned backbone, as when re-created from HF
    with torch.no_grad():
        trained.head.weight.add_(5.0)
        trained.encoder["lora_A"].weight.add_(2.0)
    state = trainable_state_dict(trained)
    load_model_state(fresh, state, scope=STATE_SCOPE_TRAINABLE)
    assert torch.equal(fresh.head.weight, trained.head.weight)
    assert torch.equal(fresh.encoder["lora_A"].weight, trained.encoder["lora_A"].weight)
    assert torch.equal(fresh.encoder["base"].weight, trained.encoder["base"].weight)
    size = lambda s: sum(v.numel() for v in s.values())  # noqa: E731
    assert size(state) < size(trained.state_dict())


def test_missing_trainable_or_unexpected_keys_are_errors():
    net = Net()
    state = trainable_state_dict(net)
    broken = {k: v for k, v in state.items() if k != "head.bias"}
    with pytest.raises(RuntimeError, match="missing"):
        load_model_state(net, broken, scope=STATE_SCOPE_TRAINABLE)
    with pytest.raises(RuntimeError, match="unexpected"):
        load_model_state(net, {**state, "ghost.weight": torch.zeros(1)}, scope=STATE_SCOPE_TRAINABLE)
    with pytest.raises(RuntimeError):  # legacy 'full' scope still demands every key
        load_model_state(net, state)


def test_resume_accepts_a_trainable_only_checkpoint(tmp_path):
    trained = Net(seed=3)
    with torch.no_grad():
        trained.head.bias.fill_(7.0)
    optimizer = torch.optim.AdamW([p for p in trained.parameters() if p.requires_grad])
    save_checkpoint(
        str(tmp_path), "latest.pt", epoch=2, global_step=10,
        model_state=trainable_state_dict(trained), optimizer_state=optimizer.state_dict(),
        scheduler_state={}, scaler_state=None, best_metric=0.5, patience_count=1,
        extra={"state_scope": STATE_SCOPE_TRAINABLE},
    )
    fresh = Net(seed=3)
    fresh_opt = torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad])
    state = find_latest_checkpoint(str(tmp_path), "latest.pt", torch.device("cpu"), fresh, fresh_opt)
    assert state.resumed and state.global_step == 10
    assert float(fresh.head.bias.item()) == 7.0
