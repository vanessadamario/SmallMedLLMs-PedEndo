model_list=("llama3.1" "bloom-7B" "qwen2-7B")
for model in "${model_list[@]}"; do
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 001
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 002
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 001 --eliminate_letter_token
done