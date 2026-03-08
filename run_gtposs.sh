python main.py --run inference --local --model gpt-oss-20b --max_tokens 1500 --prompt_template 001
python main.py --run inference --local --model gpt-oss-20b --max_tokens 1500 --prompt_template 002
python main.py --run inference --local --model gpt-oss-20b --max_tokens 1500 --prompt_template 001 --eliminate_letter_token

for temp in 0.3 0.6 1.0; do
    for i in {0..9}; do
        python main.py \
            --run inference \
            --local \
            --model gpt-oss-20b \
            --max_tokens 1500 \
            --prompt_template 001 \
            --do_sample \
            --temperature $temp
    done
done
