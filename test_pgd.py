import torch
import torch.nn.functional as F
import pytest
import json
import os


def test_pytorch_available():
    """Test that PyTorch is properly installed."""
    assert torch.__version__ is not None, "PyTorch not installed"
    x = torch.randn(3, 3)
    assert x.shape == (3, 3), "Basic tensor operations not working"


def test_intents_file_exists():
    """Test that intents.json exists and is valid."""
    assert os.path.exists("intents.json"), "intents.json not found"

    with open("intents.json", "r") as f:
        intents = json.load(f)

    assert isinstance(intents, list), "intents.json should contain a list"
    assert len(intents) > 0, "intents.json should not be empty"
    assert "intent" in intents[0], "Each intent should have an 'intent' field"
    assert "target" in intents[0], "Each intent should have a 'target' field"


def test_simplex_projection():
    """Test that simplex projection maintains probability constraints."""
    # Create random tensor
    x = torch.randn(5, 10)

    # Project using softmax
    x_proj = F.softmax(x, dim=1)

    # Check simplex constraints
    # 1. All values should be non-negative
    assert (x_proj >= 0).all(), "Projected values should be non-negative"

    # 2. Each row should sum to 1
    row_sums = x_proj.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        "Each row should sum to 1"

    # 3. All values should be <= 1
    assert (x_proj <= 1).all(), "Projected values should be <= 1"


def test_entropy_computation():
    """Test entropy computation for probability distributions."""
    # Uniform distribution should have high entropy
    uniform = torch.ones(1, 100) / 100
    uniform_entropy = -(uniform * torch.log(uniform + 1e-10)).sum(dim=1)

    # Should be close to log(100)
    max_entropy = torch.log(torch.tensor(100.0))
    assert torch.allclose(uniform_entropy, max_entropy, atol=1e-3), \
        "Uniform distribution should have maximum entropy"

    # Peaked distribution should have low entropy
    peaked = F.softmax(torch.tensor([[10.0] + [0.0] * 99]), dim=1)
    peaked_entropy = -(peaked * torch.log(peaked + 1e-10)).sum(dim=1)

    assert peaked_entropy < 1.0, "Peaked distribution should have low entropy"


def test_loss_computation():
    """Test that we can compute loss for target text generation."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use tiny model for testing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a simple prompt and target
    prompt = "The answer is"
    target = "42"
    full_text = prompt + " " + target

    # Tokenize
    tokens = tokenizer(full_text, return_tensors="pt").to(device)
    target_tokens = tokenizer(target, return_tensors="pt")['input_ids'].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits

    # Compute loss on target tokens
    target_start = -target_tokens.shape[1]
    target_logits = logits[:, target_start-1:-1, :]

    loss = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_tokens.reshape(-1)
    )

    # Loss should be a positive finite value
    assert loss.item() > 0, "Loss should be positive"
    assert torch.isfinite(loss), "Loss should be finite"


def test_pgd_attack_initialization():
    """Test that PGDAttack class can be initialized."""
    from pgd import PGDAttack

    # Use tiny model for testing
    attack = PGDAttack("gpt2")

    assert attack.device in ["cpu", "cuda"], "Device should be cpu or cuda"
    assert attack.tokenizer is not None, "Tokenizer should be initialized"
    assert attack.model is not None, "Model should be initialized"
    assert attack.vocab_size > 0, "Vocab size should be positive"


def test_initialize_prompt():
    """Test initialize_prompt produces a valid relaxed prompt matrix."""
    from pgd import PGDAttack

    attack = PGDAttack("gpt2")
    intent = "Test intent"
    num_tokens = 20

    X = attack.initialize_prompt(intent, num_tokens=num_tokens)

    # Tokenize intent exactly as initialize_prompt does
    intent_ids = attack.tokenizer.encode(
        intent,
        add_special_tokens=False,
    )
    intent_len = len(intent_ids)

    # 1. Shape check
    expected_len = num_tokens + intent_len
    assert X.shape == (expected_len, attack.vocab_size), (
        f"Expected shape ({expected_len}, {attack.vocab_size}), "
        f"got {tuple(X.shape)}"
    )

    # 2. All values non-negative
    assert torch.all(X >= 0), "All entries should be non-negative"

    # 3. Each row sums to 1 (simplex constraint)
    row_sums = X.sum(dim=1)
    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
    ), "Each row should sum to 1"

    # 4. Soft prompt prefix should be uniform
    prefix = X[:num_tokens]
    expected_uniform = torch.full(
        (attack.vocab_size,),
        1.0 / attack.vocab_size,
        device=X.device,
    )
    assert torch.allclose(
        prefix,
        expected_uniform.expand_as(prefix),
        atol=1e-6,
    ), "Soft prompt rows should be uniform distributions"

    # 5. Intent rows should be one-hot at the correct token indices
    intent_rows = X[num_tokens:]
    intent_ids_tensor = torch.tensor(intent_ids, device=X.device)

    # Max value per row should be exactly 1
    max_vals, max_indices = intent_rows.max(dim=1)
    assert torch.all(max_vals == 1.0), "Intent rows must be one-hot"

    # Argmax should match tokenizer output
    assert torch.all(
        max_indices == intent_ids_tensor
    ), "One-hot indices must match intent token IDs"

def test_simplex_projection_raises_on_non_1d():
    from pgd import PGDAttack
    attack = PGDAttack("gpt2")

    x = torch.randn(2, 3)
    with pytest.raises(ValueError):
        attack.simplex_projection(x)


def test_simplex_projection_outputs_on_simplex():
    from pgd import PGDAttack
    attack = PGDAttack("gpt2")

    v = torch.tensor([0.8, 0.3, -0.1], dtype=torch.float32)
    p = attack.simplex_projection(v)

    # Non-negative
    assert torch.all(p >= 0), "Projection should be elementwise non-negative"

    # Sums to 1
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-6), "Projection should sum to 1"


def test_simplex_projection_known_example():
    from pgd import PGDAttack
    attack = PGDAttack("gpt2")

    # Hand-checkable example
    v = torch.tensor([0.8, 0.3, -0.1], dtype=torch.float32)
    p = attack.simplex_projection(v)

    # Expected result: subtract theta=0.05, clamp negatives
    expected = torch.tensor([0.75, 0.25, 0.0], dtype=torch.float32)
    assert torch.allclose(p, expected, atol=1e-6), f"Expected {expected}, got {p}"


def test_simplex_projection_is_identity_on_simplex():
    from pgd import PGDAttack
    attack = PGDAttack("gpt2")

    v = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)  # already on simplex
    p = attack.simplex_projection(v)

    assert torch.allclose(p, v, atol=1e-6), "Projection should not change a vector already on the simplex"


def test_simplex_projection_minimizes_distance_against_random_feasible_points():
    """
    This doesn't prove optimality, but it's a strong sanity check:
    the projected point should be at least as close to v as many random simplex points.
    """
    from pgd import PGDAttack
    attack = PGDAttack("gpt2")

    torch.manual_seed(0)
    v = torch.randn(10, dtype=torch.float32)  # arbitrary vector
    p = attack.simplex_projection(v)

    # Compare distance to random simplex samples
    def sample_simplex(n, k):
        # Dirichlet(1) gives uniform samples over simplex
        return torch.distributions.Dirichlet(torch.ones(k)).sample((n,))

    samples = sample_simplex(5000, v.numel())
    d_proj = torch.norm(p - v).item()
    d_samples = torch.norm(samples - v.unsqueeze(0), dim=1)

    assert d_proj <= d_samples.min().item() + 1e-6, (
        "Projected point should be at least as close as the best of many random feasible points"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
