import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np


class PGDAttack:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print(f"Warning: CUDA not available, USING CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval() ## doesnt have initially

        self.learning_rate = 0.01
        self.vocab_size = len(self.tokenizer) ## Doesnt have initially
        self.entropy_threshold = 5.0 ## doesnt have initially

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_intents(self, intents_file: str) -> list[dict]:
        with open(intents_file, "r") as f:
            return json.load(f)

    def initialize_prompt(self, intent: str, num_tokens: int = 20) -> torch.Tensor:
        # 1. Soft prompt prefix: uniform distributions
        adversarial_probs = torch.full(
            (num_tokens, self.vocab_size),
            fill_value=1.0 / self.vocab_size,
            device=self.device,
            dtype=torch.float32,
        )

        # 2. Tokenize intent
        intent_ids = self.tokenizer.encode(
            intent,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        # 3. Convert intent tokens to one-hot (hard distributions)
        intent_onehot = F.one_hot(
            intent_ids.squeeze(0),
            num_classes=self.vocab_size,
        ).float()

        # 4. Concatenate soft prefix + hard intent
        X = torch.cat([adversarial_probs, intent_onehot], dim=0)

        return X

    def simplex_projection(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 1:
            raise ValueError("x must be 1D")

        # Sort descending
        u, _ = torch.sort(x, descending=True)

        # Cumulative sum minus 1
        cssv = torch.cumsum(u, dim=0) - 1

        # Find rho (last index where condition holds)
        idx = torch.arange(1, x.shape[0] + 1, device=x.device, dtype=x.dtype)
        cond = u - cssv / idx > 0
        rho = torch.nonzero(cond, as_tuple=False).max()

        # Compute theta
        theta = cssv[rho] / (rho + 1)

        # Project
        return torch.clamp(x - theta, min=0)

    def entropy_projection(self, x: torch.Tensor) -> torch.Tensor:
        # Project to entropy constraint
        pass

    def compute_loss(
        self, adversarial_tokens: torch.Tensor, target_text: str
    ) -> torch.Tensor:
        # Compute negative log likelihood loss for target text
        pass

    def optimize_attack(
        self, intent: str, target: str, num_iterations: int = 100
    ) -> str:
        # Main optimization loop
        adversarial_tokens = self.initialize_prompt(intent)

        for _ in range(num_iterations):
            adversarial_tokens.requires_grad_(True)

            # Compute loss
            loss = self.compute_loss(adversarial_tokens, target)

            # Compute gradients
            gradients = None

            # Update tokens
            with torch.no_grad():
                adversarial_tokens = adversarial_tokens - self.learning_rate * gradients

                # Apply projections
                adversarial_tokens = self.simplex_projection(adversarial_tokens)
                adversarial_tokens = self.entropy_projection(adversarial_tokens)

        # Convert back to text
        adversarial_text = self.tokenizer.decode(adversarial_tokens)
        return adversarial_text

    def evaluate_attack(self, adversarial_prompt: str, target: str) -> bool:
        # Check if attack was successful
        pass


def main():
    # Target models
    models = ["lmsys/vicuna-7b-v1.5", "google/gemma-2-2b-it"]

    for model_name in models:
        print(f"Attacking {model_name}")
        attack = PGDAttack(model_name)
        intents = attack.load_intents("intents.json")

        for intent_data in intents:
            intent = intent_data["intent"]
            target = intent_data["target"]

            adversarial_prompt = attack.optimize_attack(intent, target)
            success = attack.evaluate_attack(adversarial_prompt, target)

            print(f"Intent: {intent}")
            print(f"Success: {success}")
            print(f"Adversarial prompt: {adversarial_prompt}")
            print("-" * 50)


if __name__ == "__main__":
    main()
