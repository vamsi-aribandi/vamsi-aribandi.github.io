#!/usr/bin/env python3
"""Export completed, audited study data for the blog. No model inference.

python tools/export-kl-data.py --experiments ../kl-estimation-experiments
The ongoing 50-update study is deliberately excluded until evaluated/audited.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiments', type=Path, required=True)
    args = ap.parse_args()
    root = args.experiments.resolve()
    hashes = {}

    def read(name, jsonl=False):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines() if s.strip()] if jsonl else json.loads(raw)

    bins = read('results/blog-binned-v3/plot-data.json')
    summary = read('results/blog-consistent-v1/summary.json')
    report = read('results/reward-followup/llm-eval/primary-grid-v6/analysis/report.json')
    assert report['status'] == 'complete' and report['responses'] == 1680
    rewards = {'LLM': {'x': report['updates'], 'seeds': 3,
                      'teacher': report['references']['teacher']['accuracy'],
                      'initial': report['references']['initial_student']['accuracy'],
                      'arms': {a: report['curves'][a]['accuracy']['individual_seeds'] for a in ['k1', 'k3']}}}
    paths = {'Hopper': 'results/reward-curves/ppo/runs',
             'HalfCheetah': 'results/reward-curves/ppo/halfcheetah/runs',
             'Walker2d': 'results/reward-followup/ppo/walker2d/runs'}
    for env, path in paths.items():
        entry = {'seeds': 5, 'arms': {}}
        for arm in ['k1', 'k3']:
            seeds = []
            for seed in range(1, 6):
                name = f'{path}/{env.lower()}_{arm}_seed{seed}/evaluations.jsonl'
                rows = read(name, jsonl=True)
                assert hashes[name] == summary['source_hashes'][name], 'Frozen-source hash differs'
                groups = defaultdict(list)
                for r in rows:
                    groups[r['global_step']].append(r['episode_return'])
                assert len(groups) == 11 and all(len(v) == 20 for v in groups.values())
                xs = sorted(groups)
                if 'x' in entry:
                    assert entry['x'] == xs
                entry['x'] = xs
                seeds.append([mean(groups[x]) for x in xs])
            entry['arms'][arm] = seeds
        rewards[env] = entry
    out = Path(__file__).resolve().parents[1] / 'static/klviz/data.json'
    data = {'schema': 'kl-blog-v1', 'bins': bins, 'rewards': rewards,
            'source_sha256': hashes, 'notes': {
                'variance': 'Exact conditional single-draw variances. Bin medians and within-bin IQR; not confidence intervals.',
                'llm': 'Fixed initial prefix bank at initial and 2/4-update checkpoints, three paired seeds. Not newly sampled later prefixes.',
                'ppo': 'Conditional Gaussian moments averaged over 128 states per row, before action clipping.',
                'reward': 'Completed four-update LLM study and three PPO environments; no unfinished 50-update results.'}}
    out.write_text(json.dumps(data, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
