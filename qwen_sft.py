from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig

print('cuda support:',torch.cuda.is_available())

def load_distill_dataset():
    ds={'messages':[]}
    with open('r1_distill.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=json.loads(line)
            sample=[
                    {'role':'system','content':'作为家长，你负责回答孩子的问题，并给出解释。'}, 
                    {'role':'user','content': line['question']}, 
                    {'role':'assistant','content': line['answer']},
            ]
            ds['messages'].append(sample)
    return Dataset.from_dict(ds)

model_name='Qwen/Qwen2.5-3B-Instruct'
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(model_name)
dataset=load_distill_dataset()

sft_config=SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    learning_rate=5e-6,
    max_seq_length=500,
    logging_steps=1,
    save_steps=0.1,
    num_train_epochs=2,
    report_to='tensorboard',
    fp16=True,
    max_grad_norm=0.1,
    output_dir='./qwen_distill/',
)
lora_config=LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0.05,
    task_type='CAUSAL_LM',
)
trainer=SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=sft_config,
    peft_config=lora_config,
)
trainer.train()