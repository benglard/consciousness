# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb

# train_grpo.py
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str): # -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    # return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    return [2.0 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def _get_model_logps(model, tokens):
    in_tokens, out_tokens = tokens.input_ids[:, :-1], tokens.input_ids[:, 1:]
    in_attn_mask, out_attn_mask = tokens.attention_mask[:, :-1], tokens.attention_mask[:, 1:]
    out = model(in_tokens, attention_mask=in_attn_mask).logits # B, T, V
    per_token_logps = torch.nn.functional.log_softmax(out, dim=-1).gather(2, out_tokens.unsqueeze(2)).squeeze(2) # B, T
    return per_token_logps

def smol_model_predictor(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    with torch.inference_mode():
        inputs = tokenizer(responses, return_tensors="pt", padding=True, truncation=True).to(device)
        per_token_logps = _get_model_logps(smol_model, inputs)
        out_attn_mask = inputs.attention_mask[:, 1:]
        avg_logps = (per_token_logps * out_attn_mask).sum(dim=1) / out_attn_mask.sum(dim=1) # B
    return avg_logps.tolist()

big_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
# big_model_name = "Qwen/Qwen2.5-7B-Instruct"
smol_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

training_args = GRPOConfig(
    output_dir='_experiments/gist',
    learning_rate=4.2e-4, #5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    # bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, #4,
    num_generations=4, #16,
    max_prompt_length=128, #256,
    max_completion_length=128, #512,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    log_on_each_node=False,
)
peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
)
model = AutoModelForCausalLM.from_pretrained(
    big_model_name,
    # torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map=None
).to(device)

smol_model = AutoModelForCausalLM.from_pretrained(
    smol_model_name,
    # torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map=None
).to(device)
smol_model.eval()
        
tokenizer = AutoTokenizer.from_pretrained(big_model_name)
tokenizer.pad_token = tokenizer.eos_token

# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        smol_model_predictor,
    ],
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)
trainer.train()