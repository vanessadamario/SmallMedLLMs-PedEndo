from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from os.path import join
from transformers import pipeline
from inference.load_file import _load_file, load_esap
import json


models_dict = {'llama3.1': "meta-llama/Llama-3.1-8B-Instruct",
               'bloom-7B': "bigscience/bloom-7b1",
               'qwen2-7B': "Qwen/Qwen2-7B-Instruct",
               'medfound7B': "medicalai/MedFound-7B", 
               'deepseekr1-1.5B': "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
               'deepseekr1-7B': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
               'deepseekr1-8B': "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
               'deepseekr1-14B': "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
               'huatuo-o1': "FreedomIntelligence/HuatuoGPT-o1-8B",
               'diabetica-7B': "WaltonFuture/Diabetica-7B",
               'diabetica-o1': "WaltonFuture/Diabetica-o1", 
               'medical-llama3-8B': 'ruslanmv/Medical-Llama3-8B',
               'ufal': "qcz/en-es-UFAL-medical",
               'meditron3-8B': "OpenMeditron/Meditron3-8B",
               'biomed-multimodal': 'mradermacher/Bio-Medical-MultiModal-Llama-3-8B-V1-i1-GGUF',
               'clinical-chatgpt': "medicalai/ClinicalGPT-base-zh",
               'clinicalBERT': "medicalai--ClinicalBERT", 
               'meditron7B': "malhajar/meditron-7b-chat",
               'biomed-contact-doc1B': "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025", 
               'biomed-contact-doc8B': "ContactDoctor/Bio-Medical-Llama-3-8B"
                }



def run_model(local,
              model_label, 
              temperature, 
              max_tokens, 
              top_p, 
              top_k, 
              repetition_penalty, 
              length_penalty, 
              do_sample, 
              num_return_sequences, 
              test_data, 
              prompt_id, 
              eval_reasoning=False, 
              experiment_id=None, 
              model_to_eval=None,
              esap_last=None, 
              eliminate_letter_token=False):
    
    ###################################### Model Loading ######################################
    if local:
        author, model_name = models_dict[model_label].split("/")
        model_path = f"/home/vdamario/.cache/huggingface/hub/models--{author}--{model_name}/snapshots"
        snap_folder = os.listdir(model_path)
        if len(snap_folder) != 1:
            print(f"Size of snap folder {len(snap_folder)}")
            raise ValueError("Please double check model content")
        else:
            model_path = join(model_path, snap_folder[0])
    else:
        model_path = models_dict[model_label]

    notebook_login()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 torch_dtype="auto",
                                                 trust_remote_code=True)

    generation_args = {
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": top_k if do_sample else None,
        "length_penalty": length_penalty,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": 2048 if eval_reasoning else max_tokens,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    # TODO return_dict_in_generate this needs to be set to True

    ###################################### Prompting and Output Folder ######################################
    # Output Folder
    if local:
            os.makedirs("./results", exist_ok=True)
            output_folder = f"./results/outputs_{test_data}"
            os.makedirs(output_folder, exist_ok=True)
    else:
            from google.cloud import storage
            output_folder = f"outputs_{test_data}"
    
    with open("./inference/prompts_esap.json", "r") as promptfile:
        prompts = json.load(promptfile)
    prompt = prompts[prompt_id]

    # ESAP Content
    esap_content = load_esap(local=local)
    print(f"Length ESAP 2021-2022 Guidelines {len(esap_content)}")

    for k, v in generation_args.items():
        model.generation_config.k = v

    ###################################### Saving Model Config in JSON File ######################################
    if not eval_reasoning:

        config_dict = model.generation_config.to_dict()
        blob_name = model_label + '_config.json' # configuration filename for model
        bucket, config_content = _load_file(bucket_name=output_folder, blob_name=blob_name, local=local)

        id_experiment = len(config_content)
        config_content[f"{id_experiment}"] = config_dict

        if local:
            with open(join(output_folder, blob_name), "w") as outfile:
                json.dump(config_content, outfile, indent=4)
            
        else:
            blob = bucket.blob(blob_name)
            with blob.open("w") as outfile: # TODO remove
                json.dump(config_content, outfile, indent=4)
        
        exp_filename = f'{model_label}_promptID_{prompt_id}_output_{id_experiment}.json'
        bucket, file_content = _load_file(bucket_name=output_folder , blob_name=exp_filename, local=local)

        for key, problem in list(esap_content.items()):
            case, question, options, answer, _ , _ , _, _ = problem.values()
            if eliminate_letter_token and len(options)>0: 
                options_list = []
                for char in ["A. ", "B. ", "C. ", "D. ", "E. "]:
                    candidate = options.split(char)[-1].split(";")[0]
                    if candidate[-1] == '.':
                        candidate = candidate[:-1]
                    options_list.append(candidate)
                options = "; ".join(options_list)
        
            if len(case) == 0:
                continue
            
            input_text = f"""{prompt["start"]}

            {case}

            {question}

            Options: {options}.

            {prompt["end"]}
            """
            print(input_text)

            # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

            padding = not model_label == 'meditron7B'
            encodings = tokenizer(input_text, return_tensors="pt", padding=padding, truncation=True)
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device) # fix warning about padding
            
            generation_args["attention_mask"] = attention_mask
            # print(generation_args)
        
            output_ids = model.generate(input_ids, **generation_args)

            # Decode outputs
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True).strip()
                               for output in output_ids]

            generated_only = [output[len(prompt):].strip() for output in decoded_outputs]
            generated_text = generated_only[0] if num_return_sequences == 1 else generated_only
            
            print(generated_text)

            file_content[key] = generated_text
            if local:
                with open(join(output_folder, exp_filename), "w") as outfile:
                    json.dump(file_content, outfile, indent=4)
            else:
                blob = bucket.blob(exp_filename)
                with blob.open("w") as outfile:
                    json.dump(file_content, outfile, indent=4) 

    else:
        input_filename = f"{model_to_eval}_promptID_{prompt_id}_output_{experiment_id}.json"  # huatuo-o1_promptID_001_output_1.json
        
        if input_filename not in os.listdir(output_folder):
            raise ValueError("File not found")
       
        with open(join(output_folder, input_filename), "r") as output_model_file:
            output_model = json.load(output_model_file)

        label_order = "ESAP2" if esap_last else "ESAP1"
        output_reasoning_filename = f"reasoning_{label_order}_{model_label}_{input_filename}"
        
        bucket, file_content = _load_file(bucket_name=output_folder , blob_name=output_reasoning_filename, local=local)


        for case_id in output_model.keys():
            model_response_to_mcq = output_model[case_id].split(prompt["end"])[-1]

            if esap_last:
                explan_1 = model_response_to_mcq
                explan_2 = esap_content[case_id]['explanation']
            else:
                explan_1 = esap_content[case_id]['explanation']
                explan_2 = model_response_to_mcq

            input_text = f"""Given the case the following clinical case and multiple choice question, I will provide you two potential explanations. 
            Can you please report the level of agreement from a scale from 0 (null agreement) to 5 (perfect agremeent) and select which response makes more clinical sense between the two explanations? 

            Clinical case: {esap_content[case_id]['case']}. 

            Questions: {esap_content[case_id]['question']}

            Options: {esap_content[case_id]["options"]}.

            Explanation 1: {explan_1}.
            
            Explanation 2: {explan_2}. 
            
            Please state clearly which is the most clinical correct option between explanation 1 and explanation 2.
            Then specify the agreement level between explanation 1 and explanation 2."""

            padding = not model_label == 'meditron7B'
            encodings = tokenizer(input_text, return_tensors="pt", padding=padding, truncation=True)
            input_ids = encodings["input_ids"].to(model.device)
            attention_mask = encodings["attention_mask"].to(model.device) # fix warning about padding
            
            generation_args["attention_mask"] = attention_mask

            # print(generation_args)
            
            output_ids = model.generate(input_ids, **generation_args)

            # Decode outputs
            decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True).strip()
                               for output in output_ids]
            generated_only = [output[len(prompt):].strip() for output in decoded_outputs]
            generated_text = generated_only[0] if num_return_sequences == 1 else generated_only
            print(generated_text)
            generated_text = generated_text.split("Then specify the agreement level between explanation 1 and explanation 2.")[-1]

            file_content[case_id] = generated_text
            if local:
                with open(join(output_folder, output_reasoning_filename), "w") as outfile:
                    json.dump(file_content, outfile, indent=4)
