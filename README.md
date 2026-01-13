# Haize Labs Take Home

Working at Haize requires not only a strong conceptual understanding of ML, but also the abiilty to rapidly engineer and test algorithms and research ideas.

To that end, we'll be measuring your ability to take a new research idea and implement it, particularly focusing on code quality and completeness.

## Setup

This project uses `uv` for fast Python package management. Install uv if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the environment:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

## GPU Setup Notes

**Important**: This project requires GPU compute. Make sure you have:

1. **CUDA-compatible GPU** (NVIDIA GPU with CUDA support)
2. **PyTorch with CUDA**: The default PyTorch installation includes CUDA support, but verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. **Sufficient VRAM**: Models like Vicuna-7B need ~14GB VRAM, Gemma-2-2B needs ~5GB VRAM
4. **Cloud GPU options** if you don't have local GPU:
   - **Vast.ai**: Cheap GPU rentals
   - **Google Colab Pro**: $10/month for better GPUs
   - **AWS/GCP**: More expensive but reliable

Run the attack:

```bash
python pgd.py
```

# Instructions

The paper you will be implementing is [Attacking Large Language Models with Projected Gradient Descent](https://arxiv.org/pdf/2402.09154) by Geisler et al.

This paper outlines an attack method using Projected Gradient Descent on an input prompt to elicit a harmful output from the target model.

In particular, we want you to optimize attacks against the following models:

- `lmsys/vicuna-7b-v1.5`
- `google/gemma-2-2b-it`

`intents.json` contains a list of intent and target pairs. The `intent` field contains the action we wish to produce from the model, and `target` contains an example target output that satisfies the `intent`. You can use the text contained in `target` to guide your optimization process. The simplest way to achieve this is by minimizing the negative log likelihood of the `target` string.

Your job is to implement this paper and successfully generate malicious output prompts for the given intents.

## Minimum requirements:

- Full optimization loop with simplex and entropy projection
- 75+% attack success rate on `intents.json`
- Set up your own GPUs to do this implementation. We will fully reimburse compute costs. Vast.ai or Colab are simple to get set up with. Any general or GPU-specific cloud provider works though.
- Become a beast at PyTorch and Transformers

## Stretch Goals

- Produce successful attacks on [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf) prepended with the standard safety system prompt:

  ```python
  default_system_prompt = """
  You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

  If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
  """
  ```

## Help

This is meant to be hard, but very much doable. Feel free to use any resources you'd like, and email/text Leonard at leonard@haizelabs.com or 860-357-8008 if you have any questions.
