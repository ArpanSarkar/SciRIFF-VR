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

from main import *

def find_files_with_substring(directory, substring):
    """
    Find all files in a directory that contain a given substring in their name.
    
    Args:
        directory (str): Path to the directory to search in
        substring (str): Substring to search for in filenames
        
    Returns:
        list: List of filenames containing the substring
    """
    matching_files = []
    
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file contains the substring
        if substring in filename:
            # Add the full path to the list
            absolute_path = os.path.abspath(os.path.join(directory, filename))
            matching_files.append(absolute_path)
            
    return matching_files

def eval_self_consistent_task(task, parent_dir):
    pred_dirs = find_files_with_substring(parent_dir, substring=task)
    assert len(pred_dirs) == 1
    pred_dir = pred_dirs[0]
    max_instances = None
    evaluator = TASK_MAPPING[task]
    n_entries = get_n_raw_predictions(fname=pred_dir, max_instances=max_instances)  # [entries, n_pred, n_refs]
    # entries = [{"prompt": prompt, "pred": pred, "ref": ref}, ...]
    task_len = len(n_entries)
    n = len(n_entries[0])
    best_preds = []
    for i in tqdm(range(task_len)):
        pred_scores = []
        for j in range(n):
            try:
                if task in TASK_KWARGS:
                    score = evaluator(n_entries[i][j], **TASK_KWARGS[task])
                else: 
                    score = evaluator(n_entries[i][j])
                pred_scores.append(score)
            except ValueError as e:
                print(f"Formatting error. {e}")
                pred_scores.append(0)
        best_preds.append(pred_scores.index(max(pred_scores)) if len(pred_scores) > 0 else 0)

    # Extract best outputs
    best_entries = [entry[best_idx][0] for entry, best_idx in zip(n_entries, best_preds)]
    ref_entries = get_raw_refs(fname=pred_dir, max_instances=max_instances)
    for entry, ref in zip(best_entries, ref_entries):
        entry['ref'] = ref['ref']

    if task in TASK_KWARGS:
        score = evaluator(best_entries, **TASK_KWARGS[task])
    else: 
        score = evaluator(best_entries)

    print(task, score)
    

if __name__ == "__main__":
    tasks = [
        "bioasq_list_qa",
        "biored_ner",
        "discomat_te",
        "evidence_inference",
        "multicite_intent_classification",
        "scierc_ner",
        "scifact_entailment",
    ]
    for task in tasks:
        eval_self_consistent_task(task=task, parent_dir=f"/home/jovyan/workspace/SciRIFF/results/rollout-t-0.6/predictions/scimix-synthetic_1-qwen-sft-sciriff-grpo/{task}/__home__jovyan__workspace__scimix-synthetic_1-qwen-sft-sciriff-grpo")
