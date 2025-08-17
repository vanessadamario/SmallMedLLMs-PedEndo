# experiment 1 with letter
model_list=("diabetica-o1" "diabetica-7B" "meditron3-8B" "medfound7B" "huatuo-o1" "clinical-chatgpt" )
for model in "${model_list[@]}"; do
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 001
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 002
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 001 --eliminate_letter_token
done
