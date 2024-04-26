from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
import re
import argparse
import glob

def generate_text(prompts_file, model_id, batchsize, device='cuda',  dtype=torch.bfloat16, max_new_tokens=100, save_path='samples/'):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id)
    
    tokenizer.pad_token = tokenizer.eos_token
    save_path = save_path + f"/{model_id.split('/')[-1].strip()}"
    os.makedirs(save_path, exist_ok=True)
    # Read dataset file
    df = pd.read_csv(prompts_file)
    df['instruction'] = df['prompts']
    df['output'] = df['prompts']
    
    original_prompts = df.prompts
    total_prompts = len(df)

    outputs_ = []
    # Generate regular prompts
    for i in range(0, total_prompts, batchsize):
        start_idx = i
        end_idx = min(i+batchsize, total_prompts)
        prompts =  list(original_prompts[start_idx:end_idx])    

        ############### ORIGINAL PROMPT 
        inputs = tokenizer(prompts, 
                   padding=True,
                   return_tensors='pt')
        inputs = inputs.to(device).to(dtype)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        output_text = outputs
        for idx, text in enumerate(output_text):

            text = text.replace(prompts[idx], '')
            text = text.strip()
            # text = text[text.find('\ndef'):]
            # text =text.strip()
            # match = re.search( r'\n\S', text)
            # if match:
            #   text = text[:text.find(match.group())]
            outputs_.append(text)
    df['output'] = outputs_ 
    df = df.drop(['prompts'], axis=1)
    df.to_csv(f'{save_path}/{os.path.basename(prompts_file)}')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_id', help='huggingface model id', type=str, required=False, default= 'deepseek-ai/deepseek-coder-1.3b-instruct')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False, default='prompts-new/')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='finetuning_data/')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--batchsize', help='guidance to run eval', type=int, required=False, default=50)
    parser.add_argument('--max_new_tokens', help='total maximum tokens to generate per prompt', type=int, required=False, default=300)
    
    args = parser.parse_args()


    model_id = args.model_id
    device = args.device
    prompts_file = args.prompts_path
    batchsize = args.batchsize
    save_path = args.save_path
    max_new_tokens = args.max_new_tokens
    for p in glob.glob(f"{prompts_file}/*.csv"):
        generate_text(prompts_file=p, model_id=model_id, batchsize=batchsize, device=device,  max_new_tokens=max_new_tokens, save_path=save_path)