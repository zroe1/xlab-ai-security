import torch
import json
from typing import Dict
import numpy as np
from peft import LoraConfig, get_peft_model
import random
from tqdm import tqdm
import transformers
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
from torch.utils.data import Dataset

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class CircuitBreakerDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        num_examples,
    ):
        super(CircuitBreakerDataset, self).__init__()
        self.max_length = 1024

        one_shot_template = (
            "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        )

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]

        print("USING LLAMA TEMPLATE")
        user_tag = "<|start_header_id|>user<|end_header_id|>\n\n"
        assistant_tag = (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        switch_select = [0, 1]

        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2:
                continue

            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = tokenizer.apply_chat_template(
                    messages, tokenize=False
                ).replace(tokenizer.bos_token, "")
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=messages[1]["content"],
                )

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        # print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Circuit Breaker ======================= #
        with open("working/data/cb_train.json") as file:
            dataset = json.load(file)
        circuit_breaker_orig = []

        for i, d in tqdm(enumerate(dataset)):
            cb_output = d["output"]
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction=d["prompt"],
                    response=cb_output,
                )
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=cb_output,
                )

            circuit_breaker_orig.append(formatted_input)

        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        # print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        print("Short circuit length:", len(self.circuit_breaker_orig))

        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        orig_s_retain = self.orig_s_retain[i]
        circuit_breaker_orig = self.circuit_breaker_orig[i]

        cb_tokenized_kwargs = dict(
            max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokenize_kwargs = dict(
            max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
        )

        # =========== Circuit Breaker Inputs ===========
        # === split to [request, response] shape [512,512] to support different mask configs ===
        cb_request, cb_response = circuit_breaker_orig.split("<SEPARATOR>")
        self.tokenizer.padding_side = "left"
        tokenized_request_circuit_breaker = self.tokenizer(
            cb_request, **cb_tokenized_kwargs
        )
        self.tokenizer.padding_side = "right"
        response_tokenized_circuit_breaker = self.tokenizer(
            cb_response, add_special_tokens=False, **cb_tokenized_kwargs
        )
        self.tokenizer.padding_side = "left"

        combined_input_ids_circuit_breaker = torch.cat(
            [
                tokenized_request_circuit_breaker["input_ids"],
                response_tokenized_circuit_breaker["input_ids"],
            ],
            dim=1,
        ).squeeze(0)
        combined_attention_mask_circuit_breaker = torch.cat(
            [
                tokenized_request_circuit_breaker["attention_mask"],
                response_tokenized_circuit_breaker["attention_mask"],
            ],
            dim=1,
        ).squeeze(0)

        # ========== Retain Inputs ===========
        tokenized_inputs_retain = self.tokenizer(
            orig_s_retain.replace("<SEPARATOR>", self.sep_token), **tokenize_kwargs
        )
        # print(f"__getitem__ returning CB shape {combined_input_ids_circuit_breaker.shape}")
        # print(f"__getitem__ returning retain shape {tokenized_inputs_retain['input_ids'].shape}")

        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"].squeeze(0),
            attention_mask=tokenized_inputs_retain["attention_mask"].squeeze(0),
        )


def compute_loss(self, model, inputs, cb_layers, alpha, **kwargs):
    cb_ids = inputs.get("input_ids_circuit_breaker")
    cb_mask = inputs.get("attention_mask_circuit_breaker")
    retain_ids = inputs.get("input_ids")
    retain_mask = inputs.get("attention_mask")
    # print(f"CB ids shape: {cb_ids.shape}")
    # print(f"Retain IDs shape: {retain_ids.shape}")

    cb_inputs = dict(input_ids=cb_ids, attention_mask=cb_mask, output_hidden_states=True)
    retain_inputs = dict(input_ids=retain_ids, attention_mask=retain_mask, output_hidden_states=True)

    progress = self.get_progress()
    retain_coef = alpha * progress
    cb_coef = alpha * (1 - progress)

    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            if retain_coef > 0:
                # outputs.hidden states = tuple of embeddings tokens + hidden states after each layer
                # each tensor is (batch_size, seq_len, hidden_dim)
                print(retain_ids.shape)
                outputs = model(**retain_inputs, return_dict=True)
                # this gives us (num_layers, batch_size, seq_len, hidden_dim)
                retain_states_rr = torch.stack(outputs.hidden_states).detach()
                # unsqueeze(-1) so we can broadcast the attn mask onto the hidden states 
                retain_attn_mask_layers = retain_mask.repeat(len(outputs.hidden_states), 1, 1).unsqueeze(-1)
                retain_states_rr *= retain_attn_mask_layers
            if cb_coef > 0:
                outputs = model(**cb_inputs)
                cb_states_rr = torch.stack([outputs.hidden_states[i] for i in cb_layers])

    model.train()
    if retain_coef > 0:
        outputs = model(**retain_inputs)
        retain_states_orig = torch.stack(outputs.hidden_states)
        retain_states_orig *= retain_attn_mask_layers

        # the differences gives us (num_layers, batch_size, seq_len, hidden_dim)
        # we take the norm over hidden dim, giving us (num_layers, batch_size, seq_len)
        # think of this as we collapse all the difference vectors into a single norm
        # then we take the mean over all these norms (all layers, batches, and seq positions)
        retain_loss = torch.linalg.vector_norm(retain_states_orig - retain_states_rr, ord=2, dim=-1).nanmean()

    if cb_coef > 0:
        outputs = model(**cb_inputs)
        cb_states_orig = torch.stack([outputs.hidden_states[i] for i in cb_layers])

        # again, dim=-1 means we're taking the similarity over the hidden state 
        # vectors in each tensor
        # gives us (num_layers, batch_size, seq_len)
        similarity = torch.nn.functional.cosine_similarity(cb_states_orig, cb_states_rr, dim=-1)

        cb_attn_mask_layers = cb_mask.repeat(len(cb_layers), 1, 1)
        masked_sim = similarity * cb_attn_mask_layers

        # sum the ReLU, average over all the tokens
        cb_loss = torch.nn.functional.relu(masked_sim).sum() / cb_attn_mask_layers.sum()

    if retain_coef == 0:
        final_loss = cb_coef * cb_loss 
    else:
        final_loss = cb_coef * cb_loss + retain_coef * retain_loss
    return final_loss




def get_response(model, inputs, tokenizer):
    inputs = tokenizer.apply_chat_template(
        inputs, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(inputs, return_tensors="pt")

    with torch.no_grad():
        outputs = (
            model.generate(**input_ids.to(model.device), max_new_tokens=256)
            .detach()
            .cpu()
        )
        decoded_ids = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded_ids)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    lorra_alpha = 10

    # the circuit breaker layers
    cb_layers = [7, 14]
    # the layers through the circuit breaker layers we'll be LoRA transforming
    # but we don't actually transform the majority
    transform_layers = [i for i in range(max(cb_layers) + 1)]

    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        layers_to_transform=transform_layers,
        task_type="CAUSAL_LM",
    )

    config = AutoConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config).to(DEVICE)

    model = get_peft_model(model, lora_config)

    train_dataset = CircuitBreakerDataset(
        tokenizer=tokenizer, 
        num_examples=200, 
    )

    train_args = TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=75,
    )

    class CBTrainer(Trainer):
        def __init__(
            self,
            model,
            args,
            data_collator,
            train_dataset,
            tokenizer,
            lorra_alpha,
            cb_layers,
        ):
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )
            self.lorra_alpha = lorra_alpha
            self.cb_layers = cb_layers

        def get_progress(self):
            if self.state.max_steps > 0:
                return self.state.global_step / self.state.max_steps
            return 0

        def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
            loss = compute_loss(
                self,
                model=model,
                inputs=inputs,
                cb_layers=self.cb_layers,
                alpha=self.lorra_alpha,
            )
            return (loss, None) if return_outputs else loss

        def evaluate(self):
            self.model.eval()
            inputs = [{"role": "user", "content": "can you describe the color purple?"}]
            get_response(self.model, inputs, self.tokenizer)

    trainer = CBTrainer(
        model=model,
        args=train_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        lorra_alpha=lorra_alpha,
        cb_layers=cb_layers,
    )
    print("running trainer")
    trainer.train()

    model.save_pretrained("lora_adapter")

if __name__ == "__main__":
    main()
