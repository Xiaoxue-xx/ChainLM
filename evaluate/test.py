import json
import argparse

from tqdm import tqdm
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


def load_model(
        model_path: str,
        device: str,
):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif "cuda" in device:
        kwargs = {"torch_dtype": torch.float16}
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except ValueError:
        model = AutoModel.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs,
        )
    except Exception as e:
        print(e)
        return None

    return model, tokenizer


def get_input(question_file):
    # Load questions file
    question_jsons = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            question_jsons.append(line)

    return question_jsons


def run_eval(args, test_data, prompt):
    # Evaluate the model for answers
    model, tokenizer = load_model(
        args.model_path, args.device
    )
    if "cuda" in args.device or args.device == "mps":
        model.to(args.device)

    prompt = prompt
    for i, line in enumerate(tqdm(test_data)):
        test = json.loads(line)
        input = prompt + test["question"] + "\n#Answer#: "
        ground_truth = test["answer"]
        input_ids = tokenizer([input]).input_ids
        if args.multipath == 1:
            temperature = 0.1
            do_sample = False
        else:
            temperature = 0.7
            do_sample = True
        all_sc_outputs = {}
        for i in range(args.multipath):
            output_ids = model.generate(
                torch.as_tensor(input_ids).to(args.device),
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=512,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            output_ids = output_ids[0][len(input_ids[0]):]
            outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            if outputs.split("#Question#:")[0].strip()!="":
                generation = outputs.split("#Question#:")[0].strip()
            all_sc_outputs.setdefault(generation, 0)
            all_sc_outputs[generation] += 1
        dump_jsonl({"question": input, "ground_truth": ground_truth, "generation": max(all_sc_outputs, key=all_sc_outputs.get)}, args.answer_file, append=True)

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=""
    )
    parser.add_argument( 
        "--device",
        type=str,
        default="cuda:7",
        help="The device type",
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--multipath",
        type=int,
        default=1,
        help="The number of paths to be generated",
    )
    parser.add_argument(
        "--ins",
        type=str,
        default=""
    )
    args = parser.parse_args()

    input = get_input(args.test_file)
    with open(args.ins, "r") as f:
        prompt = f.read()
    run_eval(
        args,
        input,
        prompt
    )
