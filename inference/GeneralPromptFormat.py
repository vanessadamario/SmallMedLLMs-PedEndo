import os
import time
import json
import torch
#import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from accelerate import infer_auto_device_map
from huggingface_hub import notebook_login
from google.cloud import storage
from load_file import load_esap



def get_or_create_bucket(bucket_name, location1="us-central1"):
    """Check if a Google Cloud Storage bucket exists; if not, create it."""
    client = storage.Client()
    print('bucket1')
    bucket = client.bucket(bucket_name)
    print('bucket2')
    if not bucket.exists():
        print('bucket3')
        bucket.create(location=location1)  # Change location if needed
        print('bucket4')
        print(f"Bucket '{bucket_name}' created.")
    else:
        print('bucket5')
        print(f"Bucket '{bucket_name}' already exists.")
    print('bucket6')
    return bucket




tokens=800
temperature=0.1
top_p=0.9
top_k=75
repetition_penalty=1.2  #1.5
do_sample=True   #False
num_return_sequences=1

location="us-central1"
bucket_name = "delete_this_bucket_randy"
 

pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto",  # Auto-detects the best dtype, 
                device_map="auto"  # Automatically assigns the model to the GPU if available
        )
 
input_text = f"""Patient Information:
Age: 30 years old
Gender: Female
Chief Complaint: Enlarged thyroid gland with nodules found during physical examination for 1 week
Present Illness: The patient discovered thyroid abnormalities during a physical examination. An ultrasound showed "enlarged thyroid with heterogeneous echogenicity and a small cystic nodule in the right lobe." There is no associated pain or hoarseness, no fatigue, no palpitations, no edema, and no other discomfort. She is here for consultation in our department today.
Physical Examination: No exophthalmos. Thyroid gland not palpable, no nodules or tenderness detected. Heart rate 80 bpm, regular rhythm, no murmurs. Fine tremor (-), no edema in both lower limbs.
 
Using this information, define what needs to be ordered to correcrly diagnose this patient. 
Give a list of appropriate tests and a short explaination of the reason to order each test.
It needs to be concise, complete, and not exceed {tokens} tokens.
The response should **ONLY** contain the diagnostic workup in outline format. Do **not** repeat any part of the input."""
 
output = pipe(input_text,
              max_new_tokens=tokens,
              temperature=temperature, 
              top_p=top_p,
              top_k=top_k, 
              repetition_penalty=repetition_penalty, 
              do_sample=do_sample, 
              num_return_sequences=num_return_sequences)

client = storage.Client()
bucket = get_or_create_bucket(bucket_name, location)

blob_name = "Llama_prompt_output.json"
blob = bucket.blob(blob_name)

# Check if the file exists and load its content
file_content = []
if blob.exists():
    with blob.open("r") as f:
        try:
            file_content = json.load(f)
            if not isinstance(file_content, list):  # Ensure it's a list for appending
                file_content = [file_content]
        except json.JSONDecodeError:
            file_content = []  # If the file is empty or invalid, start fresh

notebook_login()

print("\nResponse\n")
generated_text = output[0]["generated_text"]
print(generated_text)

entry = {
    "input_text": input_text,
    "parameters": {
        "max_new_tokens": tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "num_return_sequences": num_return_sequences
    },
    "generated_text": generated_text
}

# Append new data
file_content.append(entry)

# Write back the updated content
print('8')
with blob.open("w") as outfile:
    json.dump(file_content, outfile, indent=4)
print('9')

"""

# Start time
start_time = time.time()
print(f"Start time: {start_time}")


model_path = "./MedFound-Llama3-8B-finetuned"
model_path = "medicalai/MedFound-176B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if model_path == "medicalai/MedFound-176B":
     # Load model in 4-bit mode to save memory
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",            # Automatically assign devices
    load_in_4bit=True,            # Enable 4-bit quantization
    llm_int8_enable_fp32_cpu_offload=True,  # Enable FP32 CPU offloading
    offload_folder="./offload_model"  # Folder to store offloaded layers
    )

    print("Model loaded successfully in 4-bit mode.")

else:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

data = pd.read_json('data/test.zip', lines=True)


_max_new_tokens = 100
_temperature = 0.7

# Try in medfound max_new_tokens = 50 and temperature = 0.0 x2 - Looking for the same output in medfound
# Try in vllm max_new_tokens 50 and temperature = 0.0 x2 - Looking for the same output in vllm
# Try in medfound 363 rows, max_new_tokens = 200, and temperature = 0.7 x1

"""
"""
with open(f"vllm_test_demo_temp_7_max_token_{_max_new_tokens}_doctor_6.json", 'w', encoding='utf-8') as f:
    f.close()

for idx in range(3):   #range(data.shape[0]):
    #input_text = f"### User:{data['context'].iloc[idx]}\n\nPlease provide a detailed and comprehensive diagnostic analysis of this medical record.\n### Assistant:"
    #input_text = f"### User:{data['context'].iloc[idx]}\nCreate a detailed and comprehensive diagnostic diagnosis. Do not display the input."
    #input_text = f"### User:{data['context'].iloc[idx]}\nYou are doctor responsible for ordering the correct tests and diagnosing the patient based on their presentation. Compose an appropriate workup, including lab work, imaging, and other tests, and give your top ten diagnoses."
    #input_text = f"### User:{data['context'].iloc[idx]}\nWhat is the differential diagnosis and what tests should I order to diagnose the patient?"
    #input_text = f"### User:{data['context'].iloc[idx]}\nCreate a differential diagnosis and tell me what tests to order to diagnose the patient. Do not duplicate text."
    #input_text = f"### User:{data['context'].iloc[idx]}\nTell me what tests to order to diagnose the patient. Do not duplicate any tests."
    #input_text = f"### User:{data['context'].iloc[idx]}\nWhat are the top four diagnoses for this patient? Do not give repeat diagnoses."
    input_text = f"### User:{data['context'].iloc[idx]}\nWhat is the diagnosis and diagnosis code or ICD-10 code. Do not repeat information."

    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to('cuda').to(torch.float16)
    output_ids = model.generate(input_ids, max_new_tokens=_max_new_tokens, temperature=_temperature, do_sample=True).to(model.device)
    #output_ids = model.generate(input_ids, max_new_tokens=_max_new_tokens, do_sample=False).to(model.device)
    generated_text = tokenizer.decode(output_ids[0,len(input_ids[0]):], skip_special_tokens=True)
    
    json_data = {}
    json_data[data['qid'].iloc[idx].astype(str)] = generated_text

    with open(f"vllm_test_demo_temp_7_max_token_{_max_new_tokens}_doctor_6.json", 'a', encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)  # write key pair objects as json formatted stream to json file
        f.write('\n')
    f.close()

    print("Generated Output:\n", generated_text)
    
    end_time = time.time()
    print(f"\n\nIndex: {idx}; Execution Time: {end_time - start_time:.2f} seconds")


# End time
end_time = time.time()
print(f"\n\nExecution Time: {end_time - start_time:.2f} seconds")"
""
"""