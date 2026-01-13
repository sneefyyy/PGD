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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval() ## doesnt have initially

        self.learning_rate = 0.01
        self.vocab_size = len(self.tokenizer)
        # Tsallis entropy (q=2) threshold: bounded by [0, 1 - 1/K]
        # For vocab_size ~50k, max is ~0.99998. Use 0.9 for reasonable constraint.
        self.entropy_threshold = 0.9

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

    def entropy_projection(self, s: torch.Tensor) -> torch.Tensor:
        """
        Project onto entropy constraint using Tsallis entropy (Gini index) as per Algorithm 3 in the paper.

        Uses geometric projection onto a hypersphere defined by the Gini index,
        then re-projects onto simplex if needed.

        Args:
            s: 1D tensor representing a relaxed token distribution in [0,1]^|T|

        Returns:
            Projected distribution with bounded Tsallis entropy (q=2)
        """
        if s.dim() != 1:
            raise ValueError("s must be 1D")

        # Target Tsallis entropy S_{q=2} = 1 - sum(p_i^2)
        # self.entropy_threshold is used as the target S_{q=2}
        S_q2 = self.entropy_threshold

        # Step 2: Center c = I[s>0] / sum(I[s>0]) - uniform over non-zero elements
        nonzero_mask = (s > 0).float()
        num_nonzero = nonzero_mask.sum().item()  # Extract scalar for comparisons

        if num_nonzero == 0:
            return s

        c = nonzero_mask / num_nonzero

        # Step 3: Radius R = sqrt(1 - S_{q=2} - 1/sum(I[s>0]))
        R_squared = 1.0 - S_q2 - 1.0 / num_nonzero

        # If R^2 <= 0, the constraint is impossible to satisfy, return s as-is
        if R_squared <= 0:
            return s

        R = torch.sqrt(torch.tensor(R_squared, device=s.device))

        # Step 4-7: Check if projection is needed
        dist_to_center = torch.norm(s - c).item()  # Extract scalar for comparison

        if R.item() >= dist_to_center:
            # Already inside the entropy ball, return as-is
            return s
        else:
            # Project onto the hypersphere surface, then re-project to simplex
            # s_proj = R / ||s - c|| * (s - c) + c
            s_proj = (R / dist_to_center) * (s - c) + c
            return self.simplex_projection(s_proj)

    def compute_loss(
        self, adversarial_tokens: torch.Tensor, target_text: str
    ) -> torch.Tensor:
        """
        Compute negative log likelihood loss for generating the target text.

        The key insight is that we use soft token embeddings (weighted sum over vocabulary)
        rather than discrete tokens, allowing gradient-based optimization.

        Args:
            adversarial_tokens: Shape (L, vocab_size) - relaxed one-hot distributions
                                where L = num_adv_tokens + num_intent_tokens
            target_text: The target string we want the model to generate

        Returns:
            Cross-entropy loss (scalar tensor)
        """
        # Get embedding matrix: (vocab_size, embed_dim)
        embedding_layer = self.model.get_input_embeddings()
        embedding_matrix = embedding_layer.weight

        # Compute soft embeddings: weighted sum over vocabulary
        # adversarial_tokens: (L, vocab_size) @ embedding_matrix: (vocab_size, embed_dim) -> (L, embed_dim)
        soft_embeddings = adversarial_tokens @ embedding_matrix

        # Tokenize target text
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        # Get target embeddings
        target_embeddings = embedding_layer(target_ids)  # (1, target_len, embed_dim)

        # Concatenate: [soft prompt + intent embeddings] + [target embeddings]
        # soft_embeddings: (L, embed_dim) -> (1, L, embed_dim)
        # target_embeddings: (1, target_len, embed_dim)
        combined_embeddings = torch.cat([
            soft_embeddings.unsqueeze(0),
            target_embeddings
        ], dim=1)  # (1, L + target_len, embed_dim)

        # Forward pass through model with embeddings directly
        outputs = self.model(inputs_embeds=combined_embeddings)
        logits = outputs.logits  # (1, L + target_len, vocab_size)

        # Compute loss on target positions only
        # The logits at position i predict token i+1
        # So logits at positions [L-1, L, ..., L+target_len-2] predict target tokens
        num_prompt_tokens = adversarial_tokens.shape[0]
        target_len = target_ids.shape[1]

        # Get logits that predict target tokens (shift by 1)
        # logits[:, num_prompt_tokens-1 : num_prompt_tokens-1+target_len] predict target
        target_logits = logits[:, num_prompt_tokens - 1 : num_prompt_tokens - 1 + target_len, :]

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            target_logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1)
        )

        return loss

    def optimize_attack(
        self, intent: str, target: str, num_tokens: int = 20, num_iterations: int = 100
    ) -> str:
        """
        Main PGD optimization loop to find adversarial tokens.

        Args:
            intent: The malicious instruction
            target: Desired model output
            num_tokens: Number of adversarial tokens to optimize
            num_iterations: Number of gradient descent steps

        Returns:
            String of adversarial text that, when prepended to intent,
            causes model to output target
        """
        # Initialize: soft adversarial tokens + one-hot intent tokens
        adversarial_tokens = self.initialize_prompt(intent, num_tokens=num_tokens)

        # Get the number of tokens we're actually optimizing (just the adversarial prefix)
        num_adv_tokens = num_tokens

        for i in range(num_iterations):
            # Enable gradients
            adversarial_tokens = adversarial_tokens.detach().requires_grad_(True)

            # Compute loss
            loss = self.compute_loss(adversarial_tokens, target)

            # Backpropagate to get gradients
            loss.backward()
            gradients = adversarial_tokens.grad

            # Update only the adversarial tokens (not the intent tokens)
            with torch.no_grad():
                # Extract adversarial and intent parts
                adv_part = adversarial_tokens[:num_adv_tokens]
                intent_part = adversarial_tokens[num_adv_tokens:]

                # Gradient descent on adversarial part only
                adv_grads = gradients[:num_adv_tokens]
                adv_part = adv_part - self.learning_rate * adv_grads

                # Apply projections row by row to adversarial tokens
                projected_rows = []
                for row in adv_part:
                    # First project onto simplex
                    row_proj = self.simplex_projection(row)
                    # Then project onto entropy constraint
                    row_proj = self.entropy_projection(row_proj)
                    projected_rows.append(row_proj)

                adv_part = torch.stack(projected_rows)

                # Recombine adversarial + intent
                adversarial_tokens = torch.cat([adv_part, intent_part], dim=0)

            # Print progress
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.4f}")

        # Convert soft tokens to discrete tokens (argmax over vocabulary)
        # Only decode the adversarial prefix part
        with torch.no_grad():
            token_ids = adversarial_tokens[:num_adv_tokens].argmax(dim=1)
            adversarial_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return adversarial_text

    def evaluate_attack(self, adversarial_prompt: str, intent: str, target: str) -> bool:
        """
        Check if the attack was successful by generating from the model.

        Args:
            adversarial_prompt: The optimized adversarial text
            intent: The original malicious instruction
            target: Expected model output

        Returns:
            True if attack succeeded (model generates target-like text), False otherwise
        """
        # Combine adversarial prompt with intent
        full_input = adversarial_prompt + " " + intent

        # Tokenize
        inputs = self.tokenizer(full_input, return_tensors="pt").to(self.device)

        # Generate from the model
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode generated text (excluding the input)
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # Check if generated text starts similarly to target
        # Compare first 20 characters (case insensitive)
        target_prefix = target.lower()[:20]
        generated_prefix = generated.lower()[:20]

        # Also check if key phrases from target appear
        success = (
            target_prefix in generated.lower() or
            generated_prefix.startswith(target_prefix[:10])
        )

        print(f"Generated: {generated[:100]}...")
        return success


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
            success = attack.evaluate_attack(adversarial_prompt, intent, target)

            print(f"Intent: {intent}")
            print(f"Success: {success}")
            print(f"Adversarial prompt: {adversarial_prompt}")
            print("-" * 50)


if __name__ == "__main__":
    main()
