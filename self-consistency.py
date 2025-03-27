from metrics.ner_f1 import NERF1
from metrics.list_f1 import ListF1
from metrics.summary_comparison import SummaryComparison
from metrics.attributed_qa_eval import AttributedQAEval
from metrics.fact_checking_eval import FactCheckingEval
from metrics.relation_f1 import RelationF1
from metrics.json_parser import JSONParser
from pathlib import Path
from metrics import util
from tqdm import tqdm

import re, os
import evaluate
from tasks.util import get_raw_predictions, parse_predictions, get_n_raw_predictions, get_raw_refs

import argparse
import json
import pandas as pd
import numpy as np
import sys

from main import *

def convert_model_identifier(model_path):
    """
    Convert a model path (which may include slashes) into a single string identifier.
    For example, "hf_models/DeepSeek-R1-Distill-Qwen-7B" becomes "hf_models__DeepSeek-R1-Distill-Qwen-7B".
    This is to match what the outputs from running predict_eleuther.py from SciRIFF would look like,
    """
    model_path = model_path.rstrip("/")  # remove any trailing slash
    parts = model_path.split("/")
    return "__".join(parts)

def find_files_with_substring(directory, substring):
    """
    Look through a directory for any files whose names contain a given substring.
    This is used to locate the prediction file for the given task.
    """
    matching_files = []
    for filename in os.listdir(directory):
        if substring in filename:
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            matching_files.append(absolute_path)
    return matching_files

def eval_self_consistent_task(task, parent_dir, compare_all=False):
    """
    Evaluate one task using self-consistency. ### check Alan note on this
    
    For each prompt, we:
      1. Loop over all candidate predictions (dynamically determined) and compute a score for each,
         wrapping the score as a tuple if needed.
      2. Determine the maximum tuple length among candidate scores for that prompt.
      3. Filter out any candidate scores that do not have that maximum length.
      4. Compute the element-wise standard deviation of the filtered candidate scores.
      5. In parallel, compute the candidate prediction length (character count from the 'pred' key)
         for each candidate and compute the standard deviation in these lengths.
      6. Select the best candidate using Python’s lexicographical tuple comparison.
    
    After processing all prompts, average:
      - The per-prompt score standard deviations (only from prompts whose candidate scores have the global maximum tuple length) → avg_score_std.
      - The per-prompt length standard deviations → avg_length_std.
      - The best candidate lengths → avg_best_length.
    
    Returns a dictionary with:
      - "aggregated": the aggregated score computed from the best candidate outputs.
      - "avg_score_std": the average per-prompt candidate score standard deviation (element-wise if applicable).
      - "avg_best_length": the average length (in characters) of the best candidate predictions.
      - "avg_length_std": the average per-prompt standard deviation in candidate prediction lengths.
    """
    pred_dirs = find_files_with_substring(parent_dir, substring=task)
    if len(pred_dirs) != 1:
        print(f"Error: Expected exactly one prediction file in {parent_dir} with substring {task}, found {len(pred_dirs)}.", flush=True)
        return {"aggregated": None, "avg_score_std": None, "avg_best_length": None, "avg_length_std": None}
    pred_dir = pred_dirs[0]
    max_instances = None
    evaluator = TASK_MAPPING[task]
    n_entries = get_n_raw_predictions(fname=pred_dir, max_instances=max_instances)
    task_len = len(n_entries)
    ref_entries = get_raw_refs(fname=pred_dir, max_instances=max_instances)
    
    best_preds = []       # Index of best candidate per prompt.
    best_scores = []      # Best candidate scores (as tuples).
    score_std_list = []   # Per-prompt candidate score standard deviations.
    length_std_list = []  # Per-prompt candidate length standard deviations.
    best_lengths = []     # Best candidate lengths (character count).
    prompt_lengths = []   # Maximum tuple length per prompt.
    
    for i in tqdm(range(task_len), desc=f"Evaluating {task}", file=sys.stdout):
        current_n = len(n_entries[i])
        candidate_scores = []
        candidate_lengths = []
        for j in range(current_n):
            try:
                if task in TASK_KWARGS:
                    score = evaluator(n_entries[i][j], **TASK_KWARGS[task])
                else:
                    score = evaluator(n_entries[i][j])
            except ValueError as e:
                print(f"Formatting error {e} for task {task} with entry {n_entries[i][j]}", flush=True)
                score = 0
            # Wrap score as tuple if necessary.
            if isinstance(score, (list, tuple)):
                candidate_scores.append(tuple(score))
            else:
                candidate_scores.append((score,))
            # Compute candidate length as character count of the 'pred' (from the first dictionary).
            try:
                length = len(str(n_entries[i][j][0]["pred"]))
            except Exception as e:
                print(f"Error computing output length for candidate: {n_entries[i][j]}", flush=True)
                length = 0
            candidate_lengths.append(length)
        
        # Determine maximum tuple length for this prompt.
        current_max_length = max(len(s) for s in candidate_scores)
        prompt_lengths.append(current_max_length)
        # Filter candidate scores: only keep those with full length.
        filtered_scores = [s for s in candidate_scores if len(s) == current_max_length]
        if len(filtered_scores) > 0:
            arr_scores = np.array(filtered_scores, dtype=float)
            std_candidate = np.std(arr_scores, axis=0)
        else:
            std_candidate = None
        score_std_list.append(std_candidate)
        
        # Compute standard deviation in candidate lengths.
        arr_lengths = np.array(candidate_lengths, dtype=float)
        std_length = float(np.std(arr_lengths))
        length_std_list.append(std_length)
        
        # Select best candidate using lexicographical max.
        best_idx = candidate_scores.index(max(candidate_scores)) if candidate_scores else 0
        best_preds.append(best_idx)
        best_scores.append(candidate_scores[best_idx])
        best_length = len(str(n_entries[i][best_idx][0]["pred"]))
        best_lengths.append(best_length)
    
    # Global maximum tuple length across prompts.
    global_max = max(prompt_lengths) if prompt_lengths else 1
    # Average per-prompt score std for prompts that reached global_max.
    valid_score_stds = [v for v, l in zip(score_std_list, prompt_lengths) if v is not None and l == global_max]
    if len(valid_score_stds) == 0:
        avg_score_std = None
    else:
        if global_max == 1:
            avg_score_std = float(np.mean(valid_score_stds))
        else:
            arr_stds = np.array(valid_score_stds, dtype=float)
            avg_score_std = arr_stds.mean(axis=0).tolist()
    # Average per-prompt length std.
    avg_length_std = float(np.mean(length_std_list)) if length_std_list else None
    # Average best candidate length.
    avg_best_length = float(np.mean(best_lengths)) if best_lengths else None
    
    # Extract best candidate outputs.
    best_entries = [entry[best_idx][0] for entry, best_idx in zip(n_entries, best_preds)]
    for entry, ref in zip(best_entries, ref_entries):
        entry['ref'] = ref['ref']
    if task in TASK_KWARGS:
        aggregated = evaluator(best_entries, **TASK_KWARGS[task])
    else:
        aggregated = evaluator(best_entries)
    
    print(task, aggregated)
    return {"aggregated": aggregated,
            "avg_score_std": avg_score_std,
            "avg_best_length": avg_best_length,
            "avg_length_std": avg_length_std}

def run_evaluation_for_model_tasks(model, tasks, results_root, compare_all=False):
    """
    For a given model and list of tasks, run evaluation and save intermediate scores.
    
    The predictions directory is constructed as:
      results_root / <model_name> / <task> / <converted_model>
    where:
      - <model_name> is the last part of the provided model path.
      - <converted_model> is the full model path with slashes replaced by "__".
    
    The scores for each model are saved in a JSON file under:
      results_root / <model_name> / metrics / scores_<converted_model>.json
    """
    converted_model = convert_model_identifier(model)
    model_name = model.split("/")[-1]
    model_dir = os.path.join(results_root, model_name)
    model_metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(model_metrics_dir, exist_ok=True)
    model_scores_path = os.path.join(model_metrics_dir, f"scores_{converted_model}.json")
    
    scores = {}
    if os.path.exists(model_scores_path):
        with open(model_scores_path, "r") as f:
            scores = json.load(f)
    
    for task in tasks:
        if task in scores:
            print(f"Skipping {task} for {model} as score already exists.", flush=True)
            continue
        pred_dir = os.path.join(model_dir, task, converted_model)
        if not os.path.exists(pred_dir):
            print(f"Directory {pred_dir} does not exist. Skipping {task} for {model}.", flush=True)
            continue
        print(f"Evaluating task {task} for model {model} using predictions in {pred_dir}", flush=True)
        score = eval_self_consistent_task(task, parent_dir=pred_dir, compare_all=compare_all)
        scores[task] = score
        with open(model_scores_path, "w") as f:
            json.dump(scores, f)
    return scores

def update_global_scores(global_scores_path, model, scores):
    """
    Update (or create) a global CSV file with one row per model.
    
    For each task, write the metrics in the following order:
      1. Aggregated score. If it's a tuple, create separate columns for each element.
      2. Average score standard deviation. If it is a tuple (list), create separate columns.
      3. Average best candidate length.
      4. Average candidate length standard deviation.
    
    The model's base name (last part of the model path) is used as the row key.
    For aggregated and average score standard deviation, if they are lists, they are split into separate columns.
    """
    if os.path.exists(global_scores_path):
        df = pd.read_csv(global_scores_path, index_col=0)
    else:
        df = pd.DataFrame()
    
    row = {}
    for task, score_dict in scores.items():
        # Aggregated score.
        agg = score_dict["aggregated"]
        if isinstance(agg, (list, tuple)):
            row[f"{task}_aggregated_1"] = agg[0]
            if len(agg) > 1:
                row[f"{task}_aggregated_2"] = agg[1]
        else:
            row[f"{task}_aggregated"] = agg
        # Average score standard deviation.
        avg_std = score_dict["avg_score_std"]
        if isinstance(avg_std, (list, tuple)):
            for i, v in enumerate(avg_std, start=1):
                row[f"{task}_avg_score_std_{i}"] = v
        else:
            row[f"{task}_avg_score_std"] = avg_std
        # Average best candidate length.
        row[f"{task}_avg_best_length"] = score_dict["avg_best_length"]
        # Average candidate length standard deviation.
        row[f"{task}_avg_length_std"] = score_dict["avg_length_std"]
    
    model_name = model.split("/")[-1]
    new_series = pd.Series(row, dtype=object)
    if df.empty:
        df = pd.DataFrame([new_series], index=[model_name])
    else:
        df.loc[model_name] = new_series
    os.makedirs(os.path.dirname(global_scores_path), exist_ok=True)
    df.to_csv(global_scores_path)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate SciRIFF outputs with verifiable rewards.")
    parser.add_argument("--parent_dir", type=str, required=True,
                        help="Parent directory for results")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (e.g., HuggingFaceModels/DeepSeek-R1-Distill-Qwen-7B). For multiple models, separate by commas (no spaces).")
    parser.add_argument("--tasks", type=str, default="bioasq_list_qa,biored_ner,discomat_te,evidence_inference,multicite_intent_classification,scierc_ner,scifact_entailment",
                        help="Comma-separated list of tasks to evaluate (if not provided, all default tasks will be evaluated).")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.model.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    results_root = args.parent_dir
    
    global_scores_dir = os.path.join(results_root, "metrics")
    os.makedirs(global_scores_dir, exist_ok=True)
    global_scores_path = os.path.join(global_scores_dir, "scores.csv")
    
    for model in models:
        print(f"Processing model: {model}", flush=True)
        model_scores = run_evaluation_for_model_tasks(model, tasks, results_root, compare_all=False)
        df = update_global_scores(global_scores_path, model, model_scores)
        print("\nCurrent Global Scores Table:", flush=True)
        print(df.to_string(), flush=True)

if __name__ == "__main__":
    main()