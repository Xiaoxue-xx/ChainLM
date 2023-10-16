#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
from typing import List, Optional, Tuple, Union
# from flash_attn.flash_attn_triton import flash_attn_func
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, apply_rotary_pos_emb

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_multi_turn_no_instruction": (
        "The following is a conversation between a human and an AI assistant namely JARVIS (a sophisticated AI assistant in the Marvel Cinematic Universe (MCU) film). "
        "JARVIS is an open-source AI assistant developed by GSAI, Renmin University of China. "
        "The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. "
        "The human will ask questions on previous conversation or new topics. The AI assistant tries not to ask questions and always provides responses in as much detail as possible. "
        "Complete the transcript in exactly that format. \n[|Human|] Hello!\n[|AI|] Hi! How can I help you?\n[|Human|] {input}\n[|AI|] "
    ),
    "prompt_multi_turn_no_instruction_v2": (
        "The following is a conversation between a human and an AI assistant namely JARVIS, developed by GSAI, Renmin University of China. "
        "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "[|Human|]:{input}\n[|AI|]:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attention: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    prompt_type: Optional[str] = field(default="instruction")
    dialog_augmentation: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    # print(trainer.args.should_save)
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, prompt_type: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        prompt_input, prompt_no_input, prompt_dialog = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_multi_turn_no_instruction"]
        prompt_dialog_v2 = PROMPT_DICT["prompt_multi_turn_no_instruction_v2"]
        self.sources, self.targets = [], []
        for path in data_path.split(','):
            if 'alpaca_data.json' in path:
                list_data_dict = utils.jload(path)

                logging.warning("Formatting inputs...")
                if prompt_type == 'instruction':
                    sources = [
                        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                        for example in list_data_dict
                    ]
                elif prompt_type == 'dialog':
                    sources = []
                    for example in list_data_dict:
                        if example.get("input", "") != "":
                            source = prompt_dialog.format_map(dict(input=f"{example['instruction'].strip()} {example['input'].strip()}"))
                        else:
                            source = prompt_dialog.format_map(dict(input=example['instruction'].strip()))
                        sources.append(source)
                elif prompt_type == 'dialog_v2':
                    sources = []
                    for example in list_data_dict:
                        if example.get("input", "") != "":
                            source = prompt_dialog_v2.format_map(dict(input=f"{example['instruction'].strip()} {example['input'].strip()}"))
                        else:
                            source = prompt_dialog_v2.format_map(dict(input=example['instruction'].strip()))
                        sources.append(source)
                targets = [f"{source}{example['output']}{tokenizer.eos_token}" for source, example in zip(sources, list_data_dict)]
                self.sources.extend(sources)
                self.targets.extend(targets)
            elif 'unified_chip2.jsonl' in path:
                with open(path, 'r') as f:
                    content = [json.loads(line)['text'] for line in f.readlines()]
                for c in content:
                    self.sources.append(c.split('\n<bot>:')[0].replace('<human>: ', ''))
                    self.targets.append(c.replace('\n<bot>: ', '').replace('<human>: ', '') + tokenizer.eos_token)
            elif 'unified_chip2_llm_response_idx' in path:
                # random.seed(42)
                with open(path, 'r') as f:
                    for line in f.readlines():
                        try:
                            # if random.random() > 0.25:
                            #     continue
                            c = json.loads(line)
                            if prompt_type == 'instruction':
                                source = prompt_no_input.format_map(dict(instruction=c['Input']))
                            elif prompt_type == 'dialog':
                                source = prompt_dialog.format_map(dict(input=c['Input']))
                            elif prompt_type == 'dialog_v2':
                                source = prompt_dialog_v2.format_map(dict(input=c['Input']))
                            self.sources.append(source)
                            self.targets.append(source + c['LLM_Gneration'] + tokenizer.eos_token)
                            # self.sources.append(c['Input'])
                            # self.targets.append(c['Input'] + c['LLM_Gneration'] + tokenizer.eos_token)
                        except:
                            pass
            elif 'sharegpt_v7' in path:
                with open(path, 'r') as f:
                    for line in f.readlines():
                        try:
                            c = json.loads(line)
                            source = prompt_no_input.format_map(dict(instruction=c['Input']))
                            self.sources.append(source)
                            self.targets.append(source + c['LLM_Generation'] + tokenizer.eos_token)
                            # self.sources.append(c['Input'])
                            # self.targets.append(c['Input'] + c['LLM_Generation'] + tokenizer.eos_token)
                        except:
                            pass
            elif 'merge_v' in path:
                with open(path, 'r') as f:
                    for line in f.readlines():
                        c = json.loads(line)
                        source = prompt_no_input.format_map(dict(instruction=c['inputs']))
                        self.sources.append(source)
                        self.targets.append(source + c['targets'] + tokenizer.eos_token)
                        # self.sources.append(c['inputs'])
                        # self.targets.append(c['inputs'] + c['targets'] + tokenizer.eos_token)
            elif 'data.jsonl' in path:
                with open(path, 'r') as f:
                    for line in f.readlines():
                        c = json.loads(line)
                        source = prompt_no_input.format_map(dict(instruction=c['prompt']))
                        self.sources.append(source)
                        self.targets.append(source + c['response'] + tokenizer.eos_token)
                        # self.sources.append(c['prompt'])
                        # self.targets.append(c['prompt'] + c['response'] + tokenizer.eos_token)
            else:
                # random.seed(42)
                with open(path, 'r') as f:
                    for i, line in enumerate(f.readlines()):
                        try:
                            c = json.loads(line)
                        except:
                            print(path)
                            print(line)
                            raise ValueError
                        # if random.random() > 0.25:
                        #     continue

                        if 'input' in c:
                            input_text = c['input']
                        elif 'Input' in c:
                            input_text = c['Input']
                        elif 'contents' in c:
                            input_text = c['contents']
                        elif 'question' in c:
                            input_text = c['question']
                        else:
                            raise ValueError('Unknown data format')
                        # input_text.replace('\n[|Human|]:', f'{}\n[|Human|]:')
                        # print(prompt_type)
                        if prompt_type == 'instruction':
                            source = prompt_no_input.format_map(dict(instruction=input_text))
                        elif prompt_type == 'dialog':
                            source = prompt_dialog.format_map(dict(input=input_text))
                        elif prompt_type == 'dialog_v2':
                            source = prompt_dialog_v2.format_map(dict(input=input_text))
                        else:
                            raise NotImplemented
                        self.sources.append(source)

                        if 'LLM_Generation' in c:
                            output_text = c['LLM_Generation']
                        elif 'Output' in c:
                            output_text = c['Output']
                        elif 'output' in c:
                            output_text = c['output']
                        elif 'answer' in c:
                            output_text = c['answer']
                        else:
                            self.targets.append(source.rstrip('\n[|AI|]:'))
                            continue
                            # raise ValueError('Unknown data format')
                        self.targets.append(source + output_text + tokenizer.eos_token)
                print(f'Loaded {len(self.sources)} examples from {path}')

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.sources[i], labels=self.targets[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    data_args: DataArguments
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(self.data_args.dialog_augmentation)
        if self.data_args.dialog_augmentation == False:
            inputs = self.tokenizer(
                text=[instance['labels'] for instance in instances],
                text_target=[instance['input_ids'] for instance in instances],
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs['input_ids'])
            labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
            labels[torch.where(inputs['labels'] != self.tokenizer.pad_token_id)] = IGNORE_INDEX
            inputs['labels'] = labels
            return inputs
        else:
            input_ids_list, labels_list = [], []
            max_length = self.tokenizer.model_max_length
            print(max_length)
            for instance in instances:
                raw_text = instance['labels'].rstrip(self.tokenizer.eos_token)
                text = []
                for i, txt in enumerate(raw_text.split('\n[|AI|]:')):
                    if i == 0:
                        text.append(txt + '\n[|AI|]:')
                    else:
                        split_txt = txt.split('\n[|Human|]:')
                        ai_txt = split_txt[0]
                        text.append(ai_txt + self.tokenizer.eos_token)
                        if len(split_txt) == 2:
                            human_txt = split_txt[1]
                            text.append('\n[|Human|]:' + human_txt + '\n[|AI|]:')
                inputs = self.tokenizer(text=text, max_length=max_length, truncation=True)
                input_ids, labels = [], []
                for i, iids in enumerate(inputs['input_ids']):
                    if i != 0:
                        iids = iids[1:]
                    input_ids.extend(iids)
                    if i % 2 == 0:
                        labels.extend([IGNORE_INDEX] * len(iids))
                    else:
                        labels.extend(iids)
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                labels = torch.tensor(labels, dtype=torch.long)
                input_ids_list.append(input_ids[:max_length])
                labels_list.append(labels[:max_length])
                assert len(input_ids_list[-1]) == len(labels_list[-1])
                # torch.set_printoptions(profile="full")
                # print(raw_text)
                # print(text)
                # print(self.tokenizer.batch_decode(input_ids))
                # print(input_ids)
                # print(labels)
                # exit(0)
                if len(input_ids_list[-1]) > 2048:
                    print(raw_text)
                    print(input_ids)
                    print(labels)
                    exit(0)
            input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
            inputs = {
                'input_ids': input_ids,
                'labels': labels,
            }
            assert input_ids.shape == labels.shape
            if input_ids.size(1) > 2048:
                print(instances)
                print(inputs)
                exit(0)
            return inputs


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, prompt_type=data_args.prompt_type, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(data_args=data_args, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# def new_llama_attention_forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     bsz, q_len, _ = hidden_states.size()

#     query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#     value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
#     # [bsz, nh, t, hd]

#     if past_key_value is not None:
#         # reuse k, v, self_attention
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     attn_output = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), None, True, None)
#     attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#     attn_output = self.o_proj(attn_output)

#     if not output_attentions:
#         attn_weights = None

#     return attn_output, attn_weights, past_key_value

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # if model_args.flash_attention:
    #     if isinstance(model, LlamaForCausalLM):
    #         LlamaAttention.forward = new_llama_attention_forward
    # model.parallelize()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    if "llama" in model_args.model_name_or_path.split('/')[-1]:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # print(data_module)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

