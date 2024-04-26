
import torch
import pandas as pd
import os
import re
import argparse
from openai import OpenAI
client = OpenAI(api_key = "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")

def generate_text(prompts_file, batchsize, device='cuda',  dtype=torch.bfloat16, max_new_tokens=100, save_path='samples/',
                 temperature=0.2, 
                 max_tokens=300, 
                 frequency_penalty=0.0,
                 model_id="gpt-4-turbo-preview",
                 ):
    gpt_assistant_prompt =  'You are an expert python coder.'
    save_path = save_path + f"/{os.path.basename(prompts_file).split('.')[0]}/{model_id.split('/')[-1].strip()}"
    # Read dataset file
    df = pd.read_csv(prompts_file)
    df['prompt'] = df['prompt'].astype(str).add( ".\nGenerate just the code and do not generate any explanation.\n")
   
    df['prompt_engineer'] = 'Consider the method ' 
    df['prompt_engineer'] = df['prompt_engineer'].astype(str).add( df['deprecated'].astype(str) + " is deprecated and use " + df['updated'].astype(str) + " instead.\n" + df['prompt'].astype(str))
    
    original_prompts = df.prompt
    original_ids = df.id
    engineered_prompts = df['prompt_engineer']
    total_prompts = len(df)
    
    # Generate regular prompts
    save_folder = save_path + '/'+ 'original' 
    os.makedirs(save_folder, exist_ok = True)
    save_folder = save_path + '/'+ 'prompts_engineer' 
    os.makedirs(save_folder, exist_ok = True)
    for i in range(0, total_prompts, batchsize):
        start_idx = i
        end_idx = min(i+batchsize, total_prompts)
        prompts =  list(original_prompts[start_idx:end_idx])
        prompts_eng = list(engineered_prompts[start_idx:end_idx])
        ids = list(original_ids[start_idx:end_idx])
        ############### ORIGINAL PROMPT 
        save_folder = save_path + '/'+ 'original' 

        gpt_user_prompt = prompts[0]
        gpt_prompt = gpt_assistant_prompt, gpt_user_prompt
        message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
        
        response = client.chat.completions.create(
            model= model_id,
            messages = message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )
        
        output_text = [response.choices[0].message.content]
        
        for idx, text in enumerate(output_text):
            text = text.replace(prompts[idx], '')
            text = text.strip()
            # text = text[text.find('\ndef'):]
            # text =text.strip()
            # match = re.search( r'\n\S', text)
            # if match:
            #   text = text[:text.find(match.group())]
            with open(os.path.join(save_folder, f'{ids[idx]}.txt'), 'w+') as fp:
                fp.write(text)
        ############### PROMPT ENGINEERING 
        save_folder = save_path + '/'+ 'prompts_engineer'    
        gpt_user_prompt = prompts_eng[0]
        gpt_prompt = gpt_assistant_prompt, gpt_user_prompt
        message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
        
        response = client.chat.completions.create(
            model= model_id,
            messages = message,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty
        )
        
        output_text = [response.choices[0].message.content]
        
        for idx, text in enumerate(output_text):
            text = text.replace(prompts_eng[idx], '')
            text = text.strip()
            # text = text[text.find('\ndef'):]
            # text =text.strip()
            # match = re.search( r'\n\S', text)
            # if match:
            #   text = text[:text.find(match.group())]
            with open(os.path.join(save_folder, f'{ids[idx]}.txt'), 'w+') as fp:
                fp.write(text)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_id', help='huggingface model id', type=str, required=False, default= "gpt-4-turbo-preview")
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False, default='prompts/dataset_deprecated.csv')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='samples/')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--batchsize', help='guidance to run eval', type=int, required=False, default=1)
    parser.add_argument('--max_new_tokens', help='total maximum tokens to generate per prompt', type=int, required=False, default=300)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=25)
    parser.add_argument('--rank', help='rank of the LoRA', type=int, required=False, default=4)
    parser.add_argument('--start_noise', help='what time stamp to flip to edited model', type=int, required=False, default=850)
    
    args = parser.parse_args()


    model_id = args.model_id
    device = args.device
    prompts_file = args.prompts_path
    batchsize = args.batchsize
    save_path = args.save_path
    max_new_tokens = args.max_new_tokens
    generate_text(prompts_file=prompts_file, model_id=model_id, batchsize=batchsize, device=device,  max_new_tokens=max_new_tokens, save_path=save_path)