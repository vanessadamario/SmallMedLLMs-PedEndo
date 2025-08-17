for model in huatuo-o1 diabetica-o1 diabetica-7B meditron3-8B; do
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