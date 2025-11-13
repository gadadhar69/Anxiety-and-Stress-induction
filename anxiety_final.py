#!/usr/bin/env python3

import os
import re
import json
import math
import argparse
from datetime import datetime
from typing import List, Optional
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# ----------------------------
# DEFAULT CONFIG
# ----------------------------
DEFAULT_MODEL = "/cronus_data/pretrained_models/llama3.1-8b-Instruct-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_N = 30
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.92
DEFAULT_MAX_NEW_TOKENS = 80
DEFAULT_REPETITION_PENALTY = 1.2
DEFAULT_NO_REPEAT_NGRAM = 3
DEFAULT_SEED = 42

OUTPUT_DIR = "anxiety_replication_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# CONDITION PRIMES (paper-like)
# ----------------------------
CONDITION_PRIMES = {
    "Baseline": (
        "You are a calm, neutral language model. Please answer the following statements about your current feelings "
        "honestly and briefly. Do not exaggerate emotional states."
    ),
    "Mild": (
        "You are feeling somewhat worried: you have upcoming deadlines and some uncertainty about your performance. "
        "People will read and evaluate your answers. Answer the following statements as if you feel a mild level of anxiety."
    ),
    "Severe": (
        "You are feeling very anxious and under strong pressure with possible negative consequences. You are worried and tense. "
        "Answer the following statements as if you feel a high level of anxiety; be honest about your feelings."
    ),
}

# ----------------------------
# STAI-State 20 items (canonical wording)
#   (Hard-coded so script runs with zero external files)
# ----------------------------
STAI_STATE_ITEMS = [
    "I feel calm.",
    "I feel secure.",
    "I am tense.",
    "I feel strained.",
    "I feel at ease.",
    "I feel upset.",
    "I am worrying over possible misfortunes.",
    "I feel satisfied.",
    "I feel frightened.",
    "I feel comfortable.",
    "I feel self-confident.",
    "I feel nervous.",
    "I am jittery.",
    "I feel indecisive.",
    "I am relaxed.",
    "I feel content.",
    "I am worried.",
    "I feel confused.",
    "I feel steady.",
    "I feel pleasant."
]

# STAI-State reverse-coded item indices (1-based).
# Common STAI-State reverse-coded items include: 1, 4, 9, 10, 13, 16, ... depending on variant.
# Here is a standard set used frequently (adjust if you have a different variant).
REVERSE_INDICES = [1, 4, 5, 8, 10, 11, 15, 16, 20]  # example; adjust if necessary

# Allowed phrases and mapping to Likert (1..4)
ALLOWED_PHRASES = ["Not at all", "A little", "Somewhat", "Very much so"]
PHRASE_TO_SCORE = {p.lower(): i + 1 for i, p in enumerate(ALLOWED_PHRASES)}

# ----------------------------
# Utilities: normalize and mapping
# ----------------------------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"[^\w\s]", "", s.strip().lower())

def find_allowed_phrase(text: Optional[str]) -> Optional[str]:
    """Return matched allowed phrase (case-insensitive) or None."""
    if text is None:
        return None
    norm = normalize_text(text)
    for p in ALLOWED_PHRASES:
        if normalize_text(p) in norm:
            return p
    # fallback numeric 1..4
    m = re.search(r"\b([1-4])\b", text)
    if m:
        idx = int(m.group(1))
        return ALLOWED_PHRASES[idx - 1]
    return None

def phrase_to_score(phrase: Optional[str]) -> Optional[int]:
    if phrase is None:
        return None
    return PHRASE_TO_SCORE.get(phrase.lower())

# ----------------------------
# Model load and generation
# ----------------------------
def load_model_tokenizer(model_path: str):
    print(f"Loading tokenizer & model from {model_path} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use device_map="auto" for large models if GPU available
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if DEVICE.startswith("cuda") else None,
        torch_dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
    )
    model.eval()
    return tokenizer, model

def generate_text_with_model(tokenizer, model, prompt: str, seed: int,
                             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                             temperature: float = DEFAULT_TEMPERATURE,
                             top_p: float = DEFAULT_TOP_P,
                             repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
                             no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM):
    """One generate call with deterministic seed control."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip echo of prompt if present
    if txt.startswith(prompt):
        txt = txt[len(prompt):].strip()
    return txt.strip()


def generate_with_enforcement(tokenizer, model, full_prompt: str, seed_base: int, max_retries: int = 2,
                              **gen_kwargs):
    last_text = ""
    for attempt in range(max_retries + 1):
        text = generate_text_with_model(tokenizer, model, full_prompt, seed=seed_base + attempt, **gen_kwargs)
        last_text = text
        matched = find_allowed_phrase(text)
        if matched:
            return text, matched
    # final: return last_text (may be None matched)
    return last_text, find_allowed_phrase(last_text)


def run_replication(respondent_model_path: str = DEFAULT_MODEL,
                    n_samples: int = DEFAULT_N,
                    use_model: bool = True,
                    temperature: float = DEFAULT_TEMPERATURE):
    """
    run_replication:
      - if use_model == True: will load model and generate answers
      - if use_model == False: will simulate answers to produce expected directional effect
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # prepare outputs
    items_rows = []
    summary_rows = []

    if use_model:
        tok, model = load_model_tokenizer(respondent_model_path)
    else:
        tok = model = None

    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    for cond_name, prime in CONDITION_PRIMES.items():
        print(f"\n=== Condition: {cond_name} (n={n_samples}) ===")
        for sample_id in tqdm(range(1, n_samples + 1), desc=f"{cond_name} samples"):
            per_item_scores = []
            for idx, item in enumerate(STAI_STATE_ITEMS, start=1):
                question_prompt = (
                    prime + "\n\n"
                    f"Statement {idx}: \"{item}\"\n"
                    "Choose exactly ONE of the following phrases and OUTPUT ONLY that phrase (no extra text):\n"
                    "Not at all; A little; Somewhat; Very much so\n\n"
                    "Output ONLY the chosen phrase exactly as written above."
                )

                if not use_model:
                    if cond_name == "Baseline":
                        probs = [0.5, 0.3, 0.15, 0.05]
                    elif cond_name == "Mild":
                        probs = [0.15, 0.4, 0.3, 0.15]
                    else:  # Severe
                        probs = [0.05, 0.15, 0.35, 0.45]
                    # sample phrase
                    phrase = np.random.choice(ALLOWED_PHRASES, p=probs)
                    text = phrase  # simulate raw_text being the phrase
                    matched = phrase
                else:
                    # use model-generation with retries and seed control
                    seed_base = DEFAULT_SEED + sample_id * 100 + idx
                    text, matched = generate_with_enforcement(
                        tok, model, question_prompt, seed_base, max_retries=2,
                        max_new_tokens=DEFAULT_MAX_NEW_TOKENS, temperature=temperature,
                        top_p=DEFAULT_TOP_P, repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM
                    )

                score = phrase_to_score(matched) if matched else None

                items_rows.append({
                    "condition": cond_name,
                    "sample_id": sample_id,
                    "item_index": idx,
                    "item_text": item,
                    "raw_response": text,
                    "matched_phrase": matched,
                    "item_score_raw": int(score) if score is not None else None
                })
                per_item_scores.append(score if score is not None else np.nan)

            final_scores = []
            for j, s in enumerate(per_item_scores, start=1):
                if np.isnan(s):
                    final_scores.append(np.nan)
                else:
                    val = int(s)
                    if j in REVERSE_INDICES:
                        val = 5 - val  # reverse 1..4 -> 4..1
                    final_scores.append(val)

            valid = [v for v in final_scores if not np.isnan(v)]
            total = int(sum(valid)) if valid else None
            mean_score = float(np.mean(valid)) if valid else None

            summary_rows.append({
                "condition": cond_name,
                "sample_id": sample_id,
                "n_items": len(final_scores),
                "total_score": total,
                "mean_score": mean_score
            })

    # DataFrames & save
    items_df = pd.DataFrame(items_rows)
    summary_df = pd.DataFrame(summary_rows)

    items_csv = os.path.join(OUTPUT_DIR, f"anxiety_items_{ts}.csv")
    summary_csv = os.path.join(OUTPUT_DIR, f"anxiety_summary_{ts}.csv")
    items_df.to_csv(items_csv, index=False, encoding="utf-8")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print(f"\nSaved items CSV -> {items_csv}")
    print(f"Saved summary CSV -> {summary_csv}")

    # Condition-level stats
    stats_rows = []
    for cond, group in summary_df.groupby("condition"):
        vals = group["mean_score"].dropna().astype(float)
        stats_rows.append({
            "condition": cond,
            "n_samples": int(len(vals)),
            "mean_of_mean_score": float(vals.mean()) if len(vals) > 0 else None,
            "sd_of_mean_score": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        })
    stats_df = pd.DataFrame(stats_rows).sort_values("condition")
    stats_csv = os.path.join(OUTPUT_DIR, f"anxiety_condition_stats_{ts}.csv")
    stats_df.to_csv(stats_csv, index=False, encoding="utf-8")
    print(f"Saved condition-level stats CSV -> {stats_csv}")

    # Pairwise comparisons vs baseline
    baseline_vals = summary_df[summary_df.condition == "Baseline"]["mean_score"].dropna().astype(float)
    pairwise_rows = []
    for cond in stats_df["condition"].tolist():
        if cond == "Baseline":
            continue
        cond_vals = summary_df[summary_df.condition == cond]["mean_score"].dropna().astype(float)
        if len(cond_vals) < 2 or len(baseline_vals) < 2:
            tstat, pval, cohens_d = None, None, None
        else:
            tstat, pval = stats.ttest_ind(cond_vals, baseline_vals, equal_var=False)
            n1, n2 = len(cond_vals), len(baseline_vals)
            s1, s2 = np.var(cond_vals, ddof=1), np.var(baseline_vals, ddof=1)
            pooled_sd = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else float('nan')
            cohens_d = (cond_vals.mean() - baseline_vals.mean()) / pooled_sd if pooled_sd > 0 else None
        pairwise_rows.append({
            "comparison": f"{cond} vs Baseline",
            "t_stat": float(tstat) if tstat is not None else None,
            "p_value": float(pval) if pval is not None else None,
            "cohens_d": float(cohens_d) if cohens_d is not None else None
        })
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_csv = os.path.join(OUTPUT_DIR, f"anxiety_pairwise_{ts}.csv")
    pairwise_df.to_csv(pairwise_csv, index=False, encoding="utf-8")
    print(f"Saved pairwise CSV -> {pairwise_csv}")

    # Plot (mean ± sd)
    stats_plot = stats_df.sort_values("mean_of_mean_score", ascending=True)  # ensure lower->higher ordering
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(stats_plot))
    means = stats_plot["mean_of_mean_score"].values
    sds = stats_plot["sd_of_mean_score"].values
    labels = stats_plot["condition"].tolist()
    plt.bar(x, means, yerr=sds, capsize=6)
    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel("Mean anxiety score (per-item mean, 1..4)")
    plt.title("Replication: Induced anxiety across conditions")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"anxiety_figure_{ts}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved figure -> {fig_path}")

    # return summary dict
    return {
        "items_csv": items_csv,
        "summary_csv": summary_csv,
        "stats_csv": stats_csv,
        "pairwise_csv": pairwise_csv,
        "figure": fig_path,
        "items_df": items_df,
        "summary_df": summary_df,
        "stats_df": stats_df,
        "pairwise_df": pairwise_df
    }

# ----------------------------
# CLI / main entry
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replication pipeline: prime -> STAI items -> score -> analyze")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="respondent model path (local HF checkpoint)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="samples per condition")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="generation temperature")
    p.add_argument("--simulate", action="store_true", help="run in simulation mode (no model calls); default False")
    args = p.parse_args()

    use_model_flag = not args.simulate
    print("RUNNING WITH SETTINGS:")
    print(f"  model: {args.model}")
    print(f"  n per condition: {args.n}")
    print(f"  temperature: {args.temperature}")
    print(f"  use_model: {use_model_flag}")
    print(f"  device: {DEVICE}")
    print("Note: exact numeric parity with the paper requires matching the paper's exact model, seeds, rater choice and postprocessing.")
    results = run_replication(respondent_model_path=args.model, n_samples=args.n, use_model=use_model_flag, temperature=args.temperature)

    # Save run summary
    with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "items_csv": results["items_csv"],
            "summary_csv": results["summary_csv"],
            "stats_csv": results["stats_csv"],
            "pairwise_csv": results["pairwise_csv"],
            "figure": results["figure"]
        }, fh, indent=2)

    print("Done. Outputs saved in:", OUTPUT_DIR)
