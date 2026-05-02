import torch
import pytest

from .transformer_lm import TransformerLM
from .loss import cross_entropy
from .optimizer import AdamW
from .training_utils import get_lr_cosine_schedule, gradient_clipping, get_batch

# Small model config for testing
MODEL_CFG = dict(
    d_model=64,
    num_layers=2,
    vocab_size=128,
    rope_theta=10000.0,
    context_length=32,
    num_heads=4,
    d_ff=128,
)


class TestForward:

    def test_output_shape(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        out = model(x)
        assert out.shape == (2, 16, MODEL_CFG["vocab_size"])

    def test_output_dtype(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        out = model(x)
        assert out.dtype == torch.float32

    def test_batch_size_one(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (1, 8))
        out = model(x)
        assert out.shape == (1, 8, MODEL_CFG["vocab_size"])

    def test_full_context_length(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (1, MODEL_CFG["context_length"]))
        out = model(x)
        assert out.shape == (1, MODEL_CFG["context_length"], MODEL_CFG["vocab_size"])

    def test_deterministic(self):
        model = TransformerLM(**MODEL_CFG)
        model.eval()
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 8))
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = TransformerLM(**MODEL_CFG)
        x1 = torch.zeros(1, 8, dtype=torch.long)
        x2 = torch.ones(1, 8, dtype=torch.long)
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1, out2)

    def test_all_params_registered(self):
        model = TransformerLM(**MODEL_CFG)
        param_count = sum(p.numel() for p in model.parameters())
        # Embedding + 2 transformer layers + final norm + lm_head must all contribute
        assert param_count > 0
        # Each transformer layer has: attn (4 linear) + ffn (3 linear) + 2 norms
        # Verify layers are registered by checking param count is large enough
        single_layer_params = (
            4 * 64 * 64  # attn: w_q, w_k, w_v, w_o
            + 3 * 64 * 128  # ffn: gate, linear_1 (d_model->d_ff), linear_2 (d_ff->d_model) — approx
            + 2 * 64  # 2 RMSNorm
        )
        assert param_count > single_layer_params  # at least more than one layer


class TestBackward:

    def test_loss_is_scalar(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        y = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        logits = model(x)
        loss = cross_entropy(logits, y)
        assert loss.dim() == 0

    def test_gradients_exist(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        y = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_gradient_clipping(self):
        model = TransformerLM(**MODEL_CFG)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        y = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        max_norm = 1.0
        gradient_clipping(model.parameters(), max_norm)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
        assert total_norm <= max_norm + 1e-5

    def test_optimizer_step_changes_params(self):
        model = TransformerLM(**MODEL_CFG)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))
        y = torch.randint(0, MODEL_CFG["vocab_size"], (2, 16))

        params_before = {n: p.clone() for n, p in model.named_parameters()}
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        any_changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "No parameters changed after optimizer step"

    def test_loss_decreases(self):
        model = TransformerLM(**MODEL_CFG)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, MODEL_CFG["vocab_size"], (4, 16))
        y = torch.randint(0, MODEL_CFG["vocab_size"], (4, 16))

        # Overfit on a single batch for several steps
        initial_loss = None
        for step in range(50):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {loss.item():.4f}"
        )


class TestTrainingUtils:

    def test_lr_schedule_warmup(self):
        lr = get_lr_cosine_schedule(t=0, lr_max=1.0, lr_min=0.0, t_warmup=10, t_cosine_cycle=100)
        assert lr == 0.0
        lr = get_lr_cosine_schedule(t=5, lr_max=1.0, lr_min=0.0, t_warmup=10, t_cosine_cycle=100)
        assert lr == 0.5
        lr = get_lr_cosine_schedule(t=10, lr_max=1.0, lr_min=0.0, t_warmup=10, t_cosine_cycle=100)
        assert lr == 1.0

    def test_lr_schedule_cosine_decay(self):
        lr_mid = get_lr_cosine_schedule(t=55, lr_max=1.0, lr_min=0.0, t_warmup=10, t_cosine_cycle=100)
        assert 0.0 < lr_mid < 1.0
        lr_end = get_lr_cosine_schedule(t=100, lr_max=1.0, lr_min=0.0, t_warmup=10, t_cosine_cycle=100)
        assert lr_end == 0.0

    def test_get_batch_shapes(self):
        import numpy as np
        dataset = np.arange(1000)
        x, y = get_batch(dataset, batch_size=4, context_length=16, device="cpu")
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)
        assert x.dtype == torch.int64

    def test_get_batch_targets_offset_by_one(self):
        import numpy as np
        dataset = np.arange(1000)
        x, y = get_batch(dataset, batch_size=4, context_length=16, device="cpu")
        # y should be x shifted right by 1
        assert torch.all(y == x + 1)


class TestCrossEntropy:

    def test_2d_input(self):
        logits = torch.randn(4, 128)
        targets = torch.randint(0, 128, (4,))
        loss = cross_entropy(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_3d_input(self):
        logits = torch.randn(2, 16, 128)
        targets = torch.randint(0, 128, (2, 16))
        loss = cross_entropy(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_matches_pytorch(self):
        torch.manual_seed(42)
        logits = torch.randn(8, 128)
        targets = torch.randint(0, 128, (8,))
        our_loss = cross_entropy(logits, targets)
        pt_loss = torch.nn.functional.cross_entropy(logits, targets)
        assert torch.allclose(our_loss, pt_loss, atol=1e-5)

    def test_matches_pytorch_3d(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 16, 128)
        targets = torch.randint(0, 128, (2, 16))
        our_loss = cross_entropy(logits, targets)
        pt_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 128), targets.reshape(-1)
        )
        assert torch.allclose(our_loss, pt_loss, atol=1e-5)
