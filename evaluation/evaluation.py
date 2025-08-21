import os
import re
import json
import argparse
import requests
from tqdm import tqdm
import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from utils import normalize_word
from openai import OpenAI


SCORE_TEMPLATES = {
    "tfq": {"miss": 0, "error": 0, "count": 0, "accuracy": 0, "incorrect": 0, 
           "f1_score": 0, "precision": 0, "recall": 0},
    "scq": {"miss": 0, "error": 0, "count": 0, "accuracy": 0, "incorrect": 0,
           "f1_score": 0, "precision": 0, "recall": 0},
    "mcq": {"miss": 0, "error": 0, "count": 0, "f1_score_macro": 0, "f1_score_micro": 0, "accuracy_score": 0},
    "oeq": {"miss": 0, "error": 0, "count": 0, "bleu_score": 0, "bleu_score_1": 0, "bleu_score_2": 0, "bleu_score_3": 0,
            "precision": 0, "recall": 0, "fmeasure": 0, "rouge1": 0, "rouge2": 0, "rougeL": 0, "rougeLsum": 0}
}


def deep_copy_scores():
    return {k: {**v} for k, v in SCORE_TEMPLATES.items()}


def get_prompt(response_answer, options, question_type):

    if question_type == "TFQ" or question_type == "SCQ":
        return f"Given the sentence: {response_answer}, and the following option list: {options}, identify the option from the list that best matches the meaning of the sentence. Only return the exact option from the list that matches best."
    
    elif question_type == "MCQ":
        return f"Given the sentence: {response_answer}, and the following option list: {options}, identify all options from the list that match the meaning of the sentence. Return all matching options from the list, and only return the exact options that match."
    
    return ""


def get_openai_response(prompt):
    
    chatanywhere_api_key = "your-api-key"

    client = OpenAI(
        api_key=chatanywhere_api_key, 
        base_url="https://api.chatanywhere.org/v1"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        import time
        time.sleep(3)
        return completion.choices[0].message.content
    
    except Exception as e:
        print(f"Error in call_async: {e}")
        return "error"


def parallel_exec(func, num_workers, num_samples, batch_size=1):
    
    class FunctionalDataset(Dataset):
        def __init__(self, func, num_samples):
            self.func = func
            self.num_samples = num_samples

        def __getitem__(self, index):
            return self.func(index)
        
        def __len__(self):
            return self.num_samples

    dataset = FunctionalDataset(func, num_samples)
    dataloader = DataLoader(
        dataset, 
        num_workers=num_workers, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: x
    )

    results = []
    for batch in tqdm(dataloader):
        results.extend(batch)

    return results


def calculate_accuracy(response_answer, gt_answer, options, question_type, prefix="tfq", scores_dict=None):
    # calculate accuracy and related metrics for TFQ/SCQ questions
    if scores_dict is None:
        scores_dict = deep_copy_scores()

    if response_answer is None:
        scores_dict[prefix]["miss"] += 1
        return scores_dict

    if ("error" in response_answer) or ("ERROR" in response_answer):
        scores_dict[prefix]["error"] += 1
        return scores_dict

    # get normalized answer if needed
    if response_answer not in options:
        print("response_answer not in options")
        prompt = get_prompt(response_answer, options, question_type)
        response_answer = get_openai_response(prompt)

    # calculate accuracy
    if gt_answer == response_answer:
        scores_dict[prefix]["accuracy"] += 1
    else:
        scores_dict[prefix]["incorrect"] += 1
    
    scores_dict[prefix]["count"] += 1
    
    # calculate additional metrics for binary classification
    all_options = sorted(list(set(options)))
    y_true = [1 if option == gt_answer else 0 for option in all_options]
    y_pred = [1 if option == response_answer else 0 for option in all_options]
    
    scores_dict[prefix]["f1_score"] += f1_score(y_true, y_pred, average='weighted', zero_division=0)
    scores_dict[prefix]["precision"] += precision_score(y_true, y_pred, average='weighted', zero_division=0)
    scores_dict[prefix]["recall"] += recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return scores_dict


def calculate_f1_score(response_answer, gt_answer, options, question_type, prefix="mcq", scores_dict=None):
    # calculate F1 score and related metrics for MCQ questions
    if scores_dict is None:
        scores_dict = deep_copy_scores()

    # ensure answers are in list format
    if isinstance(response_answer, str):
        response_answer = [response_answer]
    
    if isinstance(gt_answer, str):
        gt_answer = [gt_answer]
        
    if not response_answer:
        scores_dict[prefix]["miss"] += 1
        return scores_dict
    
    if any(err in str(ans) for ans in response_answer for err in ("error", "ERROR")):
        scores_dict[prefix]["error"] += 1
        return scores_dict
    
    # helper function to convert answers to binary vectors
    def get_binary_vector(answer):
        return [1 if option in answer else 0 for option in options]

    # normalize response if needed
    if not all(item in options for item in response_answer):
        prompt = get_prompt(response_answer, options, question_type)
        response_answer = get_openai_response(prompt)

    gt_vec = get_binary_vector(gt_answer)
    resp_vec = get_binary_vector(response_answer)

    # calculate metrics
    scores_dict[prefix]["count"] += 1
    scores_dict[prefix]["f1_score_macro"] += f1_score(gt_vec, resp_vec, average='macro')
    scores_dict[prefix]["f1_score_micro"] += f1_score(gt_vec, resp_vec, average='micro')
    scores_dict[prefix]["accuracy_score"] += accuracy_score(gt_vec, resp_vec)

    return scores_dict


def calculate_bleu_rouge_score(response_answer, gt_answer, prefix="oeq", scores_dict=None):
    # calculate BLEU and ROUGE scores for open-ended questions
    if scores_dict is None:
        scores_dict = deep_copy_scores()

    if not response_answer:
        scores_dict[prefix]["miss"] += 1
        return scores_dict

    # handle error cases
    if isinstance(response_answer, dict) or ("error" in str(response_answer)) or ("ERROR" in str(response_answer)):
        scores_dict[prefix]["error"] += 1
        return scores_dict

    # normalize inputs
    smooth_fn = SmoothingFunction().method1
    gt_text = normalize_word(gt_answer[0].lower())
    response_text = normalize_word(response_answer[0].lower())
    gt_tokens = gt_text.split()
    response_tokens = response_text.split()

    # calculate BLEU scores with different weightings
    bleu_weights = [
        (0.25, 0.25, 0.25, 0.25),   # Standard BLEU
        (1, 0, 0, 0),               # BLEU-1
        (0, 1, 0, 0),               # BLEU-2
        (0, 0, 1, 0)                # BLEU-3
    ]
    
    bleu_scores = [
        sentence_bleu([gt_tokens], response_tokens, 
                     smoothing_function=smooth_fn, weights=weights)
        for weights in bleu_weights
    ]

    # update scores dictionary
    scores_dict[prefix]["count"] += 1
    scores_dict[prefix]["bleu_score"] += bleu_scores[0]
    scores_dict[prefix]["bleu_score_1"] += bleu_scores[1]
    scores_dict[prefix]["bleu_score_2"] += bleu_scores[2]
    scores_dict[prefix]["bleu_score_3"] += bleu_scores[3]

    # calculate ROUGE-L precision/recall/fmeasure
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(response_text, gt_text)['rougeL']
    scores_dict[prefix]["precision"] += rouge_scores.precision
    scores_dict[prefix]["recall"] += rouge_scores.recall
    scores_dict[prefix]["fmeasure"] += rouge_scores.fmeasure

    # calculate other ROUGE metrics
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(predictions=[response_answer], references=[gt_answer])
    for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        scores_dict[prefix][metric] += rouge_results[metric]

    return scores_dict


def get_scores(data_path, n_jobs):
    
    # load dataset
    answer_list = []
    with open(data_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data = json.loads(line.strip())
                answer_list.append(data)

    views = set()
    for sample in answer_list:
        view = sample.get("metadata", {}).get("view", "Unknown").lower()
        views.add(view)
    print(f"Found {len(views)} unique views: {', '.join(views)}")

    print(f"Processing {len(answer_list)} samples...")
    sample_results = []

    def job(index):
        sample = answer_list[index]
        options = sample["options"]
        gt_answer = sample["gt_answer"]
        question_type = sample["question_type"]
        response_answer = sample["answer_info"]["response"]

        view = sample.get("metadata", {}).get("view", "Unknown").lower()
        
        scores_dict_single = deep_copy_scores()
        
        if question_type == "TFQ":
            scores_dict_single = calculate_accuracy(
                response_answer, gt_answer, options, question_type, 
                prefix="tfq", scores_dict=scores_dict_single
            )
        elif question_type == "SCQ":
            scores_dict_single = calculate_accuracy(
                response_answer, gt_answer, options, question_type, 
                prefix="scq", scores_dict=scores_dict_single
            )
        elif question_type == "MCQ":
            scores_dict_single = calculate_f1_score(
                response_answer, gt_answer, options, question_type, 
                prefix="mcq", scores_dict=scores_dict_single
            )
        elif question_type == "OEQ":
            scores_dict_single = calculate_bleu_rouge_score(
                response_answer, gt_answer, 
                prefix="oeq", scores_dict=scores_dict_single
            )
        
        return {"view": view, "scores": scores_dict_single}
    
    sample_results = parallel_exec(job, num_workers=n_jobs, num_samples=len(answer_list), batch_size=n_jobs)
    
    results_by_view = defaultdict(lambda: deep_copy_scores())
    for result in sample_results:
        view = result["view"]
        scores = result["scores"]
        
        for question_type, metrics in scores.items():
            for metric, value in metrics.items():
                results_by_view[view][question_type][metric] += value
    
    mean_scores_by_view = {}
    for view, scores in results_by_view.items():
        mean_scores_by_view[view] = {}
        
        for question_type, metrics in scores.items():
            count = metrics["count"]
            if count == 0:
                continue
                
            mean_scores_by_view[view][question_type] = {}
            for metric, value in metrics.items():
                if metric not in ["miss", "error", "count"]:
                    mean_scores_by_view[view][question_type][metric] = value / count
                else:
                    mean_scores_by_view[view][question_type][metric] = value
    
    return mean_scores_by_view


def main(args):

    print(f'Evaluating {args.data_path} ...')
    dir_path = os.path.dirname(args.data_path)
    model_name = os.path.basename(args.data_path)
    output_path = os.path.join(dir_path, model_name.replace("responses.jsonl", "scores.json"))

    # calculate scores 
    scores_by_view = get_scores(args.data_path, args.n_jobs)

    with open(output_path, 'w') as json_file:
        json.dump(scores_by_view, json_file, indent=4)
        
    print(f"Scores saved to: {output_path}")
    print("------ Evaluation completed ------")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate model responses on FaceBench benchmark')
    parser.add_argument('--data-path', type=str, default='responses.jsonl', help='Path to the JSONL file containing model responses')
    parser.add_argument('--n-jobs', type=int, default=16, help='Number of parallel workers for processing')
    args = parser.parse_args()

    main(args)