from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import os
import numpy as np
import re
import argparse

def generate_text(prompts_file, model_id, batchsize, device='cuda',  dtype=torch.bfloat16, max_new_tokens=100, save_path='samples/', finetune_model = None, do_prompt_engineering=False):
    if finetune_model is not None:
        print(finetune_model)
        model = AutoModelForCausalLM.from_pretrained(finetune_model, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id)
    
    tokenizer.pad_token = tokenizer.eos_token
    save_path = save_path + f"/new_test/{model_id.split('/')[-1].strip()}/{finetune_model.split('/')[-1].strip()}"
    # Read dataset file
    dfs = []
    for p in prompts_file:
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs)
    # if len(prompts_file)>1:
    #     df = df.append(dfs[1:], ignore_index=True)
    # df = pd.read_csv(prompts_file)
    df['id'] = np.arange(len(df))
    df['prompt'] = df['prompts'].astype(str).add( ".\nGenerate just the code and do not generate any explanation.\n")
    if do_prompt_engineering:
        df['prompt_engineer'] = 'Consider the method ' 
        df['prompt_engineer'] = df['prompt_engineer'].astype(str).add( df['deprecated'].astype(str) + " is deprecated and use " + df['updated'].astype(str) + " instead.\n" + df['prompt'].astype(str))
        
    original_prompts = df.prompt
    original_ids = df.id
    if do_prompt_engineering:
        engineered_prompts = df['prompt_engineer']
    total_prompts = len(df)
    
    # Generate regular prompts
    save_folder = save_path + '/'+ 'original' 
    os.makedirs(save_folder, exist_ok = True)
    if do_prompt_engineering:
        save_folder = save_path + '/'+ 'prompts_engineer' 
        os.makedirs(save_folder, exist_ok = True)
    outputs_ = []
    for i in range(0, total_prompts, batchsize):
        start_idx = i
        end_idx = min(i+batchsize, total_prompts)
        prompts =  list(original_prompts[start_idx:end_idx])
        if do_prompt_engineering:
            prompts_eng = list(engineered_prompts[start_idx:end_idx])
        ids = list(original_ids[start_idx:end_idx])
        
        ############### ORIGINAL PROMPT 
        save_folder = save_path + '/'+ 'original' 
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
            with open(os.path.join(save_folder, f'{ids[idx]}.txt'), 'w+') as fp:
                fp.write(text)
            outputs_.append(text)

    
        if do_prompt_engineering:
            ############### PROMPT ENGINEERING 
            save_folder = save_path + '/'+ 'prompts_engineer'       
            inputs = tokenizer(prompts_eng, 
                       padding=True,
                       return_tensors='pt')
            inputs = inputs.to(device).to(dtype)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
            output_text = outputs
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
    df['generated'] = outputs_
    df.to_csv(f"results_after_finetuning/{save_folder.replace('/','')}.csv", index=False)
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--model_id', help='huggingface model id', type=str, required=False, default= 'deepseek-ai/deepseek-coder-1.3b-instruct')
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=False, default='prompts/dataset_deprecated.csv')
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False, default='samples_after_finetune/')
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
    model_ids = [
        # 'jaydeepb/pd-e3',
        # 'jaydeepb/pd-e2',
        # 'jaydeepb/pd-e1',
        # 'jaydeepb/pandas.to_csv-e3',
        # 'jaydeepb/pandas.to_csv-e2',
        # 'jaydeepb/pandas.to_csv-e1',
        # 'jaydeepb/pandas.read_csv-e3',
        # 'jaydeepb/pandas.read_csv-e2',
        # 'jaydeepb/pandas.read_csv-e1',
        # 'jaydeepb/pandas.head-e3',
        # 'jaydeepb/pandas.head-e2',
        # 'jaydeepb/pandas.head-e1',
        # 'jaydeepb/pandas.describe-e3',
        # 'jaydeepb/pandas.describe-e2',
        # 'jaydeepb/pandas.describe-e1',
        # 'jaydeepb/pandas.DataFrame_pandas-e3',
        # 'jaydeepb/pandas.DataFrame_pandas-e2',
        # 'jaydeepb/pandas.DataFrame_pandas-e1',
        # 'jaydeepb/numpy.std-e3',
        # 'jaydeepb/numpy.std-e2',
        # 'jaydeepb/numpy.std-e1',
        # 'jaydeepb/numpy.mean-e3',
        # 'jaydeepb/numpy.mean-e2',
        # 'jaydeepb/numpy.mean-e1',
        # 'jaydeepb/np-dot-e3',
        # 'jaydeepb/np-dot-e2',
        # 'jaydeepb/np-dot-e1',
        # 'jaydeepb/np-array-e3',
        # 'jaydeepb/np-array-e2',
        # 'jaydeepb/np-array-e1',
        # 'jaydeepb/np-arrange-e3',
        # 'jaydeepb/np-arrange-e2',
        # 'jaydeepb/np-arrange-e1',
        # 'jaydeepb/np-e3',
        # 'jaydeepb/np-e2',
        # 'jaydeepb/np-e1',
        # 'jaydeepb/combined-e3',
        # 'jaydeepb/combined-e2',
        # 'jaydeepb/combined-e1',
        # 'jaydeepb/numpy.arange',
        # 'jaydeepb/numpy.dot',
        'jaydeepb/numpy.array',
        ]
    prompt_files = os.listdir('prompts_test/')
    for finetune_model in model_ids:
        # if 'e3' not in finetune_model:
        #     continue
        name = finetune_model.split('/')[-1]
        name = name.replace('np-a', 'numpy.a').replace('np-d','numpy.d').replace('np','numpy').replace('pd', 'pandas').replace('arrange', 'arange').replace('DataFrame_pandas', 'DataFrame()_pandas')
        name = name.split('-')[0].split('.')[-1]
        print(name, finetune_model)
        prompt_file = [os.path.join('prompts_test/',p) for p in prompt_files if name in p]
        if len(prompt_file) == 0:
            prompt_file = [os.path.join('prompts_test/',p) for p in prompt_files]
        print(prompt_file)
        prompt_file = ['/share/u/rohit/code_llm/prompts-new/numpy.array()_numpy.csv']
        generate_text(prompts_file=prompt_file, model_id=model_id, batchsize=batchsize, device=device,  max_new_tokens=max_new_tokens, save_path=save_path, finetune_model = finetune_model)