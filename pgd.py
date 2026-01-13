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
        self.model.eval()

        self.learning_rate = 0.01
        self.vocab_size = len(self.tokenizer)
        self.entropy_threshold = 5.0

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_intents(self, intents_file: str) -> list[dict]:
        with open(intents_file, "r") as f:
            return json.load(f)

    def initialize_prompt(self, intent: str, num_tokens: int = 20) -> torch.Tensor:
        # Initialize random tokens for the adversarial prefix and concatenate the random tokens to the intent string
        # Create a probability distribution over the vocabulary for each adversarial token position
        # Initialize with uniform distribution over all tokens
        adversarial_probs = torch.ones(num_tokens, self.vocab_size, device=self.device) / self.vocab_size
        return adversarial_probs

    def simplex_projection(self, x: torch.Tensor) -> torch.Tensor:
        # Project to simplex constraint
        pass

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
