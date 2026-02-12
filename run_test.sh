model_list=("huatuo-o1" "meditron3-8B" )
for model in "${model_list[@]}"; do
    python main.py --run inference --local --model $model --max_tokens 1500 --prompt_template 001