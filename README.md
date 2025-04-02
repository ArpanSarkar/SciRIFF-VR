# SciRIFF-VR

SciRIFF verifiable rewards.

Evaluator mapping see `main.py`. 

**Note** that `summ_comparison` and `attributed_eval` involve llm_judge and may require major revision to adapt to your workflow. The code was redundant and written for OpenAI batch jobs. 

`attributed_eval` and `fact_checking` have tuple output of 2 scores, others have 1. 

# Self-Consistency Evaluator for SciRIFF Outputs

The `self-consistency.py` script evaluates model outputs using self-consistency. It then computes:

- **Aggregated Score:** The evaluation score for each task using only the best entries.
- **Average Score Standard Deviation:** The average standard deviation of candidate scores per instance.
- **Average Best Candidate Length:** The average character count of the best candidate predictions.
- **Average Length Standard Deviation:** The average standard deviation in candidate prediction lengths per instance.

## How to Run

Use the following command (adjust the paths as needed):

```bash
python self-consistency.py \
  --parent_dir <parent_dir> \
  --model DeepSeek-R1-Distill-Qwen-7B
```

Default tasks evaluated:
bioasq_list_qa, biored_ner, discomat_te, evidence_inference, multicite_intent_classification, scierc_ner, scifact_entailment

A global TSV is generated at `/<parent_dir>/metrics/scores.tsv`.

**Note** that parsing of model outputs for each task assumes formatting from `predict_eleuther.py` from the SciRIFF repo