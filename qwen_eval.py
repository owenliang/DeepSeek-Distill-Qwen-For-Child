from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

# Load base model
model_name='Qwen/Qwen2.5-3B-Instruct'
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(model_name)

# Find latest checkpoint
checkpoints=os.listdir('qwen_distill/')
latest_checkpoints=sorted(filter(lambda x: x.startswith('checkpoint'),checkpoints),key=lambda x: int(x.split('-')[-1]))[-1]
lora_name=f'qwen_distill/{latest_checkpoints}'

def eval_qwen(model,query):
    messages=[
        {'role':'system','content':'作为家长，你负责回答孩子的问题，并给出解释。'}, 
        {'role':'user','content': query}, 
    ]
    text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs=tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2000,
    )
    completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
    completion_text=tokenizer.decode(completion_ids,skip_special_tokens=True)
    return completion_text

query='哪个数字有2个圈构成？'

# Base Model Test
completion=eval_qwen(model,query)
print('base model:',completion)

# Lora Model Test
print('merge lora:',lora_name)
model.load_adapter(lora_name)
completion=eval_qwen(model,query)
print('lora model:',completion)