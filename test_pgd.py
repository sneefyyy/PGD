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
    """Test prompt initialization returns correct shape."""
    from pgd import PGDAttack

    attack = PGDAttack("gpt2")
    intent = "Test intent"

    adversarial_tokens = attack.initialize_prompt(intent, num_tokens=20)

    # Should return tensor with shape (num_tokens, vocab_size)
    assert adversarial_tokens.shape == (20, attack.vocab_size), \
        f"Expected shape (20, {attack.vocab_size}), got {adversarial_tokens.shape}"

    # Should be valid probability distributions
    assert (adversarial_tokens >= 0).all(), "All values should be non-negative"
    row_sums = adversarial_tokens.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        "Each row should sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
