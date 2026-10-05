#!/usr/bin/env python3
"""How often is a sampled token much less likely under the policy than under
the reference? Samples responses from a policy model and scores every sampled
token under the policy and a reference model, in float32.

Two pairs, both with the Qwen2.5 tokenizer:
  distillation: student Qwen2.5-0.5B-Instruct, teacher Qwen2.5-1.5B-Instruct
  post-training: policy Qwen2.5-0.5B-Instruct, reference Qwen2.5-0.5B (base)

  pip install torch transformers numpy
  python tools/kl-token-ratios.py   # writes static/klviz/tokens.json
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

POLICY = 'Qwen/Qwen2.5-0.5B-Instruct'
REFERENCES = {'distillation': 'Qwen/Qwen2.5-1.5B-Instruct', 'post-training': 'Qwen/Qwen2.5-0.5B'}
PROMPTS = [
    'A train leaves at 3pm going 60 mph. Another leaves the same station at 4pm going 80 mph. When does the second catch up?',
    'What is the sum of all integers from 1 to 200 that are divisible by 3 or 5?',
    'Solve for x: 3x^2 - 12x + 9 = 0. Show your steps.',
    'A rectangle has perimeter 40 and area 96. What are its side lengths?',
    'How many ways can 5 people sit around a round table?',
    'If a fair die is rolled three times, what is the probability that the sum is 10?',
    'Find the remainder when 7^100 is divided by 13.',
    'A store discounts an item by 20% and then by another 15%. What is the total discount?',
    'Write a Python function that returns the n-th Fibonacci number using memoization.',
    'Write a Python function to check whether a string is a palindrome, ignoring punctuation.',
    'Explain the difference between a list and a tuple in Python, with examples.',
    'Write a SQL query that returns the top 3 customers by total order value.',
    'Implement binary search in JavaScript and explain its time complexity.',
    'What does the `yield` keyword do in Python? Give an example.',
    'Write a bash one-liner that counts the lines in all .py files in a directory tree.',
    'Explain recursion to a beginner programmer.',
    'Summarize the causes of the French Revolution in a short paragraph.',
    'Why is the sky blue? Explain for a ten-year-old.',
    'Give three tips for writing a clear email to a busy manager.',
    'What are the main differences between mitosis and meiosis?',
    'Describe how a bill becomes a law in the United States.',
    'Write a short poem about autumn leaves.',
    'What is the capital of Australia, and why is it not Sydney?',
    'Explain what inflation is and one way central banks try to control it.',
    'Translate into French: "The weather is nice today, so we are going to the park."',
    'Write a haiku about the ocean.',
    'List four renewable energy sources and one drawback of each.',
    'What is the Pythagorean theorem? Give a real-world use.',
    'Explain gradient descent in two or three sentences.',
    'Tell me a short story about a robot who learns to paint.',
    'What is the difference between weather and climate?',
    'Give a recipe for a simple tomato pasta sauce.',
]


def token_logprobs(model, ids, start, mask):
    """Log-probabilities of the response tokens ids[:, start:], zero where masked."""
    with torch.no_grad():
        logits = model(ids).logits[:, start - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(2, ids[:, start:, None])[..., 0]
    return lp[mask]


def main(samples_per_prompt=4, max_new_tokens=256, seed=0):
    torch.manual_seed(seed)
    device = os.environ.get('KL_DEVICE') or ('mps' if torch.backends.mps.is_available() else 'cpu')
    load = lambda name: AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()
    tok = AutoTokenizer.from_pretrained(POLICY)
    policy = load(POLICY)
    # Sampled responses are cached, so an interrupted run resumes where it stopped.
    cache = Path(__file__).resolve().parent / f'.kl-token-samples-{samples_per_prompt}-{max_new_tokens}.pt'
    sequences = [tuple(t.to(device) if torch.is_tensor(t) else t for t in s)
                 for s in (torch.load(cache) if cache.exists() else [])]   # (ids, prompt length, response mask)
    for prompt in PROMPTS[len(sequences):]:
        text = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').input_ids.to(device)
        out = policy.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                              num_return_sequences=samples_per_prompt, max_new_tokens=max_new_tokens,
                              pad_token_id=tok.pad_token_id or tok.eos_token_id)
        response = out[:, ids.shape[1]:]
        # Keep tokens up to and including the first end-of-turn token.
        done = (response == tok.eos_token_id) | (response == tok.pad_token_id)
        mask = torch.cumsum(done.int(), 1) - done.int() == 0
        sequences.append((out, ids.shape[1], mask))
        torch.save([(o.cpu(), n, m.cpu()) for o, n, m in sequences], cache)
        print(f'sampled {len(sequences)}/{len(PROMPTS)} prompts', flush=True)
    policy_lp = [token_logprobs(policy, s, n, m) for s, n, m in sequences]
    del policy
    result = dict(policy=POLICY, prompts=len(PROMPTS), samples_per_prompt=samples_per_prompt,
                  max_new_tokens=max_new_tokens, pairs={})
    edges = np.arange(-30, 10.01, .5)
    for pair, name in REFERENCES.items():
        reference = load(name)
        log_ratio = torch.cat([lp - token_logprobs(reference, s, n, m) for lp, (s, n, m) in zip(policy_lp, sequences)]).cpu().numpy()
        del reference
        counts, _ = np.histogram(np.clip(log_ratio, edges[0], edges[-1] - 1e-9), edges)
        below = {str(t): float((log_ratio < t).mean()) for t in [-2.6, -5.6, -8, -8.9, -17.2]}
        result['pairs'][pair] = dict(reference=name, tokens=int(log_ratio.size), kl=float(log_ratio.mean()),
                                     min=float(log_ratio.min()), below=below, edges=edges.tolist(), counts=counts.tolist())
        print(pair, result['pairs'][pair]['tokens'], 'tokens; mean log p/q', round(log_ratio.mean(), 3),
              'min', round(log_ratio.min(), 2), 'fraction below', below)
    path = Path(__file__).resolve().parents[1] / 'static/klviz/tokens.json'
    path.write_text(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    import sys
    main(*map(int, sys.argv[1:]))
