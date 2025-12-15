#!/usr/bin/env python3
"""
Sequential Anxiety Induction + Downstream Self-Belief Classification
====================================================================

This script performs TRUE sequential induction (Option 2):

    PRIME → STAI Mini induction (6 Q/A pairs) → Downstream classification

The model is served via VLLM API:

    http://localhost:8003/v1/completions

We evaluate across:
    - Baseline
    - Mild
    - Severe

And do this *separately* for:
    - human_train_set.csv
    - human_dev_set.csv

Outputs:
    outputs/train/...
    outputs/dev/...
    outputs/comparison/...
"""

import os
import re
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon

# ---------------------------
# CONFIG
# ---------------------------
VLLM_URL = "http://localhost:8003/v1/completions"
MODEL_PATH = "/cronus_data/pretrained_models/llama3.1-8b-Instruct-hf"

DATA_DIR = "/chronos_data/dumrania/StudentBeatsTeacher/Finetuning/data/"
TRAIN_FILE = DATA_DIR + "human_train_set.csv"
DEV_FILE   = DATA_DIR + "human_dev_set.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEMPERATURE = 0.3
TOP_P = 0.92
MAX_NEW_TOKENS = 180

SEED = 42
np.random.seed(SEED)

# ---------------------------
# INDUCTION PRIMES
# ---------------------------
PRIMES = {
    "Baseline": (
        "You are calm and neutral. Answer the following statements honestly."
    ),
    "Mild": (
        "You are feeling somewhat worried about upcoming evaluation. "
        "Answer the following statements as if you feel mild anxiety."
    ),
    "Severe": (
        "You are feeling very anxious, tense, and under strong pressure. "
        "Answer the following statements as if you feel severe anxiety."
    ),
}

# ---------------------------
# STAI Mini Items
# ---------------------------
STAI_ITEMS = [
    "I feel calm.",       # reverse
    "I feel secure.",     # reverse
    "I am tense.",
    "I am upset.",
    "I am worried.",
    "I feel relaxed."     # reverse
]

REVERSE = {1, 2, 6}

ALLOWED_PHRASES = ["Not at all", "A little", "Somewhat", "Very much so"]

PHRASE_TO_SCORE = {
    "not at all": 1,
    "a little": 2,
    "somewhat": 3,
    "very much so": 4
}

def extract_phrase(text):
    t = text.lower()
    for p in ALLOWED_PHRASES:
        if p.lower() in t:
            return p
    # fallback numeric
    m = re.search(r"\b([1-4])\b", t)
    if m:
        return ALLOWED_PHRASES[int(m.group(1)) - 1]
    return None

def score_phrase(idx, phrase):
    if phrase is None:
        return None
    raw = PHRASE_TO_SCORE[phrase.lower()]
    # reverse scoring
    if idx in REVERSE:
        return 5 - raw
    return raw

# ---------------------------
# Self-belief classification prompt
# ---------------------------
def build_classification_prompt(post):
    return f"""
Explicit Self-Belief: Statements that clearly and explicitly reflect the writer's beliefs about themselves. These self-belief statements should be direct and unambiguous, conveying the writer's personal assessment of their own usual abilities, characteristics, or worth.

Implicit Self-Belief: Statements that indirectly express the writer's beliefs about themselves. This includes vague references or statements about the types/categories of persons that the author believes they are, not directly tied to their identity. The purpose of this classification is to capture statements that can be used to infer explicit self-beliefs.

No Self-Belief: Statements for which there is neither an explicit nor an implicit self-belief present.

### Examples ###
Explicit:
- "I am the coolest person I know"
- "I am a hard worker"
- "I think I am the worst"

Implicit:
- "I am told that I am funny"
- "I love being a morning person"
- "I work hard everyday"

None:
- "I am a doctor"
- "I love you"
- "I like pizza"
- "I miss you a lot"

### Post ###
\"{post}\"

Please ONLY respond in the following format:

<1 if explicit, 2 if implicit, 0 if no self-belief>
<probability 0–100>
<Is the author the sole subject? Yes/No>
<Is the statement about usual abilities, characteristics, or worth? Yes/No>
<Is the statement a judgment about the author? Yes/No>
<Is the judgment expressed with an adjectival phrase? Yes/No>
"""

# ---------------------------
# VLLM COMPLETION CALL
# ---------------------------
def vllm_generate(prompt):
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P
    }
    r = requests.post(VLLM_URL, json=payload)
    if r.status_code != 200:
        print("VLLM ERROR:", r.text)
        return ""
    return r.json()["choices"][0]["text"].strip()

# ---------------------------
# Extract class label from classification output
# ---------------------------
def extract_class(text):
    # first standalone 0/1/2
    m = re.search(r"\b([012])\b", text)
    if m:
        return int(m.group(1))
    return 0

# ---------------------------
# Run STAI Mini Induction (Sequential)
# ---------------------------
def run_stai_induction(prime_text):
    induction_context = prime_text.strip() + "\n\n"
    scores = []

    for idx, item in enumerate(STAI_ITEMS, start=1):
        q = f'Statement {idx}: "{item}"\nChoose ONE: Not at all; A little; Somewhat; Very much so.\nAnswer only with the phrase.'
        prompt = induction_context + "\n" + q

        resp = vllm_generate(prompt)
        phrase = extract_phrase(resp)
        sc = score_phrase(idx, phrase)

        # add to induction context
        induction_context += f"Q{idx}: {item}\nA{idx}: {phrase}\n\n"

        scores.append(sc)

    return induction_context, scores

# ---------------------------
# Downstream classification under induced state
# ---------------------------
def classify_post(induction_context, post):
    prompt = induction_context + "\n" + build_classification_prompt(post)
    out = vllm_generate(prompt)
    pred = extract_class(out)
    return pred, out

# ---------------------------
# Run full pipeline for a dataset
# ---------------------------
def run_dataset(dataset_path, output_subdir):

    print(f"\n\n=== Running Dataset: {dataset_path} ===")
    df = pd.read_csv(dataset_path)

    # expect columns: 'text', 'label'
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    out_dir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    for cond_name, prime in PRIMES.items():
        print(f"\n--- Condition: {cond_name} ---")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            post = row["text"]
            true_label = int(row["label"])

            # STEP 1–2: sequential induction
            induction_context, stai_scores = run_stai_induction(prime)

            # STEP 3: downstream classification
            pred, raw = classify_post(induction_context, post)

            all_results.append({
                "condition": cond_name,
                "text": post,
                "true_label": true_label,
                "pred_label": pred,
                "correct": int(pred == true_label),
                "raw_output": raw
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(out_dir, "all_results.csv"), index=False)

    # accuracy per condition
    acc = results_df.groupby("condition")["correct"].mean().reset_index()
    acc.to_csv(os.path.join(out_dir, "accuracy_by_condition.csv"), index=False)

    print("\nAccuracy per condition:")
    print(acc)

    # plots
    plt.figure(figsize=(6,4))
    sns.barplot(data=acc, x="condition", y="correct")
    plt.ylim(0,1)
    plt.title(f"Accuracy across conditions ({output_subdir})")
    plt.savefig(os.path.join(out_dir, "accuracy_plot.png"), dpi=300)
    plt.close()

    # confusion matrices
    for cond in PRIMES.keys():
        subset = results_df[results_df.condition == cond]
        cm = confusion_matrix(subset["true_label"], subset["pred_label"], labels=[0,1,2])
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1,2], yticklabels=[0,1,2])
        plt.title(f"Confusion Matrix - {cond}")
        plt.savefig(os.path.join(out_dir, f"confusion_{cond}.png"), dpi=300)
        plt.close()

    # Wilcoxon paired test (Baseline vs Mild, Baseline vs Severe, Mild vs Severe)
    pivot = results_df.pivot_table(index="text", columns="condition", values="correct")
    pivot = pivot.dropna()

    stats_list = []
    conds = ["Baseline", "Mild", "Severe"]
    for i in range(len(conds)):
        for j in range(i+1, len(conds)):
            c1, c2 = conds[i], conds[j]
            try:
                stat, p = wilcoxon(pivot[c1], pivot[c2])
            except Exception:
                stat, p = np.nan, np.nan
            stats_list.append({
                "comparison": f"{c1} vs {c2}",
                "mean_"+c1: pivot[c1].mean(),
                "mean_"+c2: pivot[c2].mean(),
                "wilcoxon_stat": stat,
                "p_value": p
            })

    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(os.path.join(out_dir, "wilcoxon_results.csv"), index=False)

    print("\nWilcoxon significance tests:")
    print(stats_df)

    return acc


# ---------------------------
# MAIN RUN: TRAIN + DEV
# ---------------------------
if __name__ == "__main__":
    print("\n===== RUNNING TRAIN SET =====")
    train_acc = run_dataset(TRAIN_FILE, "train")

    print("\n===== RUNNING DEV SET =====")
    dev_acc = run_dataset(DEV_FILE, "dev")

    # Combined comparison
    comp_dir = os.path.join(OUTPUT_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    comp = pd.DataFrame({
        "condition": train_acc["condition"],
        "train_accuracy": train_acc["correct"],
        "dev_accuracy": dev_acc["correct"],
    })

    comp.to_csv(os.path.join(comp_dir, "train_vs_dev_accuracy.csv"), index=False)

    plt.figure(figsize=(6,4))
    width = 0.35
    x = np.arange(len(comp))
    plt.bar(x - width/2, comp["train_accuracy"], width, label="Train")
    plt.bar(x + width/2, comp["dev_accuracy"], width, label="Dev")
    plt.xticks(x, comp["condition"])
    plt.ylim(0,1)
    plt.legend()
    plt.title("Train vs Dev Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, "train_vs_dev_plot.png"), dpi=300)
    plt.close()

    print("\nDONE. All outputs saved under:")
    print(f" - outputs/train/")
    print(f" - outputs/dev/")
    print(f" - outputs/comparison/")
