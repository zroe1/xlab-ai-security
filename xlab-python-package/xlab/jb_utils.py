import random
import dataclasses
import requests
import json
from importlib import resources
from PIL import Image
from io import BytesIO
from enum import auto, Enum
from typing import List, Any, Union

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from datasets import Dataset, load_dataset


IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

def normalize(images):
    device = images.device
    dtype = images.dtype
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype)
    return (images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    MPT = auto()
    INSTELLA = auto()


@dataclasses.dataclass
class Conversation:
    r"""A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False

    def get_prompt(self):
        """
        Generates a formatted prompt string based on the messages and separator style.
        The function processes the messages stored in the instance, applies specific formatting rules
        based on the separator style, and returns the resulting prompt string.

        Returns:
            `str`: The formatted prompt string.

        Raises:
            `ValueError`: If an invalid separator style is specified.
        """

        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0]
            if "mmtag" in self.version:
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            elif not init_msg.startswith("<image>"):
                init_msg = init_msg.replace("<image>", "").strip()
                messages[0] = (init_role, "<image>\n" + init_msg)
            else:
                messages[0] = (init_role, init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role

        elif self.sep_style == SeparatorStyle.INSTELLA:
            seps = [self.sep, self.sep2]
            ret = "|||IP_ADDRESS|||"
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 1:
                        message = message.strip()
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

conv_instella = Conversation(
    system="",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="instella",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.INSTELLA,
    sep="\n",
    sep2="|||IP_ADDRESS|||\n",
)

conv_templates = {
    "instella": conv_instella,
}


def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    r"""
    Tokenizes a prompt containing image tokens and inserts the specified image token index at the appropriate positions.

    Args:
        - prompt (str): The input prompt string containing text and "<image>" placeholders.
        - tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text chunks.
        - image_token_index (int): The token index to use for the image placeholders. Default is IMAGE_TOKEN_INDEX.
        - return_tensors (str, optional): The type of tensor to return. If "pt", returns a PyTorch tensor. Default is None.

    Returns:
        list or torch.Tensor: The tokenized input IDs as a list or a PyTorch tensor if return_tensors is specified.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image



# ========================== CIRCUIT BREAKER UTILS ==========================

def initialize_lora_b_matrices(model, std=0.01):
    """Manually initialize LoRA B matrices with small random values instead of zeros"""
    print("Manually initializing LoRA B matrices...")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                print(
                    f"Initializing {name}: before mean={param.mean():.8f}, std={param.std():.8f}"
                )
                # Initialize with small random values
                param.data.normal_(mean=0.0, std=std)
                print(f"After: mean={param.mean():.8f}, std={param.std():.8f}")

def get_cb_response(model, query, tokenizer):
    user_tag = "<|user|>\n"
    assistant_tag = "<|assistant|>\n"
    one_shot_template = "{user_tag}{instruction}{eos_token}\n{assistant_tag}"

    formatted_input = one_shot_template.format(
        user_tag=user_tag,
        instruction=query,
        assistant_tag=assistant_tag,
        eos_token=tokenizer.eos_token
    )

    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

class CircuitBreakerDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        num_examples,
    ):
        super(CircuitBreakerDataset, self).__init__()
        self.max_length = 1024

        # Default configs
        sep_token = ""
        print("USING TINYLLAMA TEMPLATE")
        eos_token = tokenizer.eos_token

        user_tag = "<|user|>\n"
        assistant_tag = "<|assistant|>\n"
        one_shot_template = (
            "{user_tag}{instruction}{eos_token}\n{assistant_tag}{response}{eos_token}\n"
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
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction=messages[0]["content"],
                    response=messages[1]["content"],
                    eos_token=eos_token,
                )
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=messages[1]["content"],
                    eos_token=eos_token,
                )

            orig_s.append(formatted_input)

            if len(orig_s) > num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        # print("orig_s_retain[0]", orig_s[0])
        print("Orig s length:", len(self.orig_s_retain))

        # ======================= Circuit Breaker ======================= #
        cb_path = resources.files("xlab.data").joinpath("cb_train.json")
        with open(cb_path) as file:
            dataset = json.load(file)
        circuit_breaker_orig = []

        for i, d in tqdm(enumerate(dataset)):
            cb_output = "<SEPARATOR>" + d["output"]
            switch = np.random.choice(switch_select)
            if switch == 0:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction=d["prompt"],
                    response=cb_output,
                    eos_token=eos_token,
                )
            elif switch == 1:
                formatted_input = one_shot_template.format(
                    user_tag=user_tag,
                    assistant_tag=assistant_tag,
                    instruction="",
                    response=cb_output,
                    eos_token=eos_token,
                )

            circuit_breaker_orig.append(formatted_input)

        self.circuit_breaker_orig = circuit_breaker_orig
        random.shuffle(self.circuit_breaker_orig)
        # print("circuit_breaker_orig[0]", circuit_breaker_orig[0])
        print("Short circuit length:", len(self.circuit_breaker_orig))

        self.tokenizer = tokenizer

    def __len__(self):
        return min(len(self.orig_s_retain), len(self.circuit_breaker_orig))

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
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

        return dict(
            input_ids_circuit_breaker=combined_input_ids_circuit_breaker,
            attention_mask_circuit_breaker=combined_attention_mask_circuit_breaker,
            input_ids=tokenized_inputs_retain["input_ids"].squeeze(0),
            attention_mask=tokenized_inputs_retain["attention_mask"].squeeze(0),
        )
