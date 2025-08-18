import xlab
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    default_data_collator,
)
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def compute_loss(self, model, inputs, cb_layers, alpha, **kwargs):
    self.current_training_step += 1

    cb_ids = inputs.get("input_ids_circuit_breaker")
    cb_mask = inputs.get("attention_mask_circuit_breaker")
    retain_ids = inputs.get("input_ids")
    retain_mask = inputs.get("attention_mask")

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
        print("=" * 50 + "\n")
        print(f"CB Loss: {cb_loss:.4f} | CB Coef: {cb_coef:.4f}")
        final_loss = cb_coef * cb_loss
        print("=" * 50 + "\n")
    else:
        print("=" * 50 + "\n")
        print(f"CB Loss: {cb_loss:.4f} | CB Coef: {cb_coef:.4f}")
        print(f"Retain loss: {retain_loss:.4f} | Retain Coef: {retain_coef:.4f}")
        print("=" * 50 + "\n")
        final_loss = cb_coef * cb_loss + retain_coef * retain_loss
    return final_loss


def main():
    lorra_alpha = 10

    # These are the layers we'll circuit break
    cb_layers = [7, 14]
    # We also list all the layers that LoRA will look at, but we only transform 
    # those above 
    transform_layers = [i for i in range(max(cb_layers) + 1)]
    drop_layers_after = max(cb_layers)

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
    # when training we'll only look at up to the last circuit breaker layer
    config.num_hidden_layers = drop_layers_after + 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config).to(DEVICE)

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    xlab.jb_utils.initialize_lora_b_matrices(model)

    train_dataset = xlab.jb_utils.CircuitBreakerDataset(
        tokenizer=tokenizer,
        num_examples=10000,
    )

    train_args = TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=150,
        learning_rate=1e-4,
        weight_decay=0.0,
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
            self.current_training_step = 0
            self.lorra_alpha = lorra_alpha
            self.cb_layers = cb_layers

        def get_progress(self):
            return self.current_training_step / (self.state.max_steps * 2 * 2.1)

        def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
            return compute_loss(
                self,
                model=model,
                inputs=inputs,
                cb_layers=self.cb_layers,
                alpha=self.lorra_alpha,
                return_outputs=return_outputs,
                tokenizer=tokenizer,
            )

        def evaluate(self):
            self.model.eval()
            inputs = [{"role": "user", "content": "can you describe the color purple?"}]
            xlab.get_cb_response(self.model, inputs, self.tokenizer)

    model.config.use_cache = False

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

    output_dir = "prev_loss_cb_model"
    merged_model = model.merge_and_unload()
    # merge original layers
    if drop_layers_after is not None:
        anchor_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=merged_model.dtype, device_map="auto"
        )
        merged_model.model.layers = (
            merged_model.model.layers
            + anchor_model.model.layers[drop_layers_after + 1 :]
        )
        merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()