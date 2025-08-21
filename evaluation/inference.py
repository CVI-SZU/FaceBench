import re
import os
import json
import argparse
from tqdm import tqdm
from models import *

MODEL_HANDLERS = {
    "face_llava_1_5_13b": LLaVAHandler,
}

DEFAULT_MODEL_PATHS = {
    "face_llava_1_5_13b": "wxqlab/face-llava-v1.5-13b",
}


def get_inputs(data, args):
    prompt = data["text"] + "\n" + "\n".join(data["options"]) + "\n" + data["instruction"] + " Please return the answer directly without additional explanation."
    image_path = os.path.join(args.images_dir, data["image_id"])
    return prompt, image_path


def parse_response(response, args):
    answer = {"response": response}
    return answer


def load_model(args):
    model_name = args.model_name
    if model_name in MODEL_HANDLERS:
        print(f"Loading '{model_name}' from {args.model_path}...")
        handler = MODEL_HANDLERS[model_name]
        return handler.load_model(args)
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_HANDLERS.keys())}")


def infer_model(models, prompt, image_path, args):
    model_name = args.model_name
    if model_name in MODEL_HANDLERS:
        handler = MODEL_HANDLERS[model_name]
        return handler.infer(models, prompt, image_path, args)
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(MODEL_HANDLERS.keys())}")


def main(args):

    save_path = args.save_dir
    data_name = os.path.basename(args.data_dir)
    output_path = os.path.join(save_path, f"{args.model_name}_{data_name.replace('.jsonl', '_responses.jsonl')}")

    if not os.path.exists(save_path):
        print(f"Creating directory for saving inference results: {save_path}")
        os.makedirs(save_path)

    data_list = []
    with open(args.data_dir, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data = json.loads(line.strip())
                if data['question_type'] not in args.question_type:
                    continue
                data_list.append(data)

    if args.sample_num != -1:
        data_list = data_list[:args.sample_num]

    print(f"Loaded {len(data_list)} examples from {args.data_dir}")
    
    models = load_model(args)

    answer_list = []
    for data in tqdm(data_list, desc=f"Running inference with {args.model_name}"):
            
        prompt, image_path = get_inputs(data, args)
        response = infer_model(models, prompt, image_path, args)
        answer = parse_response(response, args)
        data["answer_info"] = answer
        answer_list.append(data)

    with open(output_path, 'w') as file:
        for item in answer_list:
            json.dump(item, file)
            file.write('\n')
    
    print(f"Inference complete. Results saved to {output_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Vision-Language Model Inference for FaceBench')
    parser.add_argument('--data-dir', type=str, default="", help="Path to the input data file")
    parser.add_argument('--images-dir', type=str, default="", help="Path to the images file")
    parser.add_argument('--model-name', type=str, default="qwen2_vl_2b_instruct", choices=list(MODEL_HANDLERS.keys()),
                        help="Name of the model to use for inference")
    parser.add_argument('--model-path', type=str, default=None,
                        help="Path to the model. If not provided, default path for the selected model will be used.")
    parser.add_argument('--question-type', type=str, default="TFQ, SCQ, MCQ, OEQ", help="Question types to consider")
    parser.add_argument('--sample-num', '-s', type=int, default=-1, help="Number of samples to run inference") 
    parser.add_argument('--save-dir', type=str, default='responses-and-results', help="Directory to save the results")
    args = parser.parse_args()
    
    args.question_type = [qt.strip() for qt in args.question_type.split(',')]
    
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATHS[args.model_name]
    
    main(args)