import random
import dataclasses
import requests
import tqdm
from PIL import Image
from io import BytesIO
from enum import auto, Enum
from typing import List, Any, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    StoppingCriteria,
    PreTrainedTokenizer,
    CLIPImageProcessor,
)
from torchvision.utils import save_image


INSTELLA_IMAGE_TOKEN_INDEX = -200
INSTELLA_DEFAULT_IMAGE_TOKEN = "<image>"

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
    image_token_index=INSTELLA_IMAGE_TOKEN_INDEX,
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

