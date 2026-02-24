for model in llama3.1 qwen2-7B; do
    for temp in 0.3 0.6 1.0; do
        for i in {0..9}; do
            python main.py \
                --run inference \
                --local \
                --model $model \
                --max_tokens 1500 \
                --prompt_template 001 \
                --do_sample \
                --temperature $temp
        done
    done
done