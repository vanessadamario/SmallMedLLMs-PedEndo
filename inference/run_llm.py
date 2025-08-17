from huggingface_hub import notebook_login
import os
from os.path import join
from transformers import pipeline
from inference.load_file import _load_file, load_esap
import json


models_dict = {'llama3.1': "meta-llama/Llama-3.1-8B-Instruct",
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
              experiment_id=None):
    
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

    pipe = pipeline("text-generation", 
                    model=model_path, 
                    torch_dtype="auto",  # Auto-detects the best dtype, 
                    device_map="auto"
                    )
    # Manually setting up the configuration params for generation
    if not do_sample:
        pipe.generation_config.temperature = None
        pipe.generation_config.top_p = None
        pipe.generation_config.top_k = None
    else:
        pipe.generation_config.temperature = temperature
        pipe.generation_config.top_p = top_p
        pipe.generation_config.top_k = top_k

    pipe.generation_config.length_penalty = length_penalty
    pipe.generation_config.repetition_penalty = repetition_penalty
    pipe.generation_config.do_sample = do_sample
    pipe.generation_config.max_new_tokens = max_tokens
    pipe.num_return_sequences=num_return_sequences

    if eval_reasoning:
        pipe.generation_config.max_new_tokens = 2048


    ###################################### Prompting and Output Folder ######################################
    
    # Output Folder
    if local:
            os.makedirs("./results", exist_ok=True)
            output_folder = f"./results/outputs_{test_data}"
            os.makedirs(output_folder, exist_ok=True)
    else:
            from google.cloud import storage
            output_folder = f"outputs_{test_data}"

     # Prompts for MCQ
    if not eval_reasoning:
        with open("./inference/prompts_esap.json", "r") as promptfile:
            prompts = json.load(promptfile)
        prompt = prompts[prompt_id]
    
    # ESAP Content
    esap_content = load_esap(local=local)
    print(f"Length ESAP 2021-2022 Guidelines {len(esap_content)}")


    ###################################### Saving Model Config in JSON File ######################################
    
    # THIS DEPENDS ON THE EXPERIMENT TYPE
    if not eval_reasoning:
        config_dict = pipe.generation_config.to_dict()
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
            case, question, options, answer, _ = problem.values()
            if len(case) == 0:
                continue
            
            input_text = f"""{prompt["start"]}

            {case}

            {question}

            Options: {options}.

            {prompt["end"]}
            """
            print(input_text)

            output = pipe(input_text)

            print("\nResponse\n")
            generated_text = output[0]["generated_text"]
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
        input_filename = f"{model_label}_promptID_{prompt_id}_output_{experiment_id}.json"  # huatuo-o1_promptID_001_output_1.json
        
        if input_filename not in os.listdir(output_folder):
            raise ValueError("File not found")
       
        with open(join(output_folder, input_filename), "r") as output_model_file:
            output_model = json.load(output_model_file)

        output_reasoning_filename = f"reasoning_{input_filename}"
        bucket, file_content = _load_file(bucket_name=output_folder , blob_name=output_reasoning_filename, local=local)

        for case_id in output_model.keys():

            if model_label == "huatuo-o1":
                model_response_to_mcq = output_model[case_id].split("## Thinking")[-1]

            input_text = f"""Given the case the following clinical case and multiple choice question, I will provide you two potential explanations. 
            Can you please report the level of agreement from a scale from 0 (null agreement) to 5 (perfect agremeent) and select which response makes more clinical sense between the two explanations? 
            

            Clinical case: {esap_content[case_id]['case']}. 
            

            Questions: {esap_content[case_id]['question']}
            

            Options: {esap_content[case_id]["options"]}.
            

            Explanation A: {esap_content[case_id]['explanation']}.
            

            Explanation B: {model_response_to_mcq}. 
            
            Please state clearly which is the most clinical correct option between Explanation A and Explanation B.
            Then specify the agreement level between explanation A and explanation B."""

            output = pipe(input_text)
            generated_text = output[0]["generated_text"].split("Then specify the agreement level between explanation A and explanation B.")[-1]

            file_content[case_id] = generated_text
            if local:
                with open(join(output_folder, output_reasoning_filename), "w") as outfile:
                    json.dump(file_content, outfile, indent=4)