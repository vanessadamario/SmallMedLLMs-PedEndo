# huatuo-o1
for i in {0..9}; do
    python main.py --run inference --local --model huatuo-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 1.
done

for i in {0..9}; do
    python main.py --run inference --local --model huatuo-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.6
done

for i in {0..9}; do
    python main.py --run inference --local --model huatuo-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.3
done

##############################################################
# diabetica-o1
for i in {0..9}; do
    python main.py --run inference --local --model diabetica-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 1.
done

for i in {0..9}; do
    python main.py --run inference --local --model diabetica-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.6
done

for i in {0..9}; do
    python main.py --run inference --local --model diabetica-o1 --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.3
done

##############################################################
# meditron3-8B
for i in {0..9}; do
    python main.py --run inference --local --model meditron3-8B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 1.
done

for i in {0..9}; do
    python main.py --run inference --local --model meditron3-8B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.6
done

for i in {0..9}; do
    python main.py --run inference --local --model meditron3-8B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.3
done

##############################################################
# diabetica-7B
for i in {0..9}; do
    python main.py --run inference --local --model diabetica-7B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 1.
done

for i in {0..9}; do
    python main.py --run inference --local --model diabetica-7B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.6
done

for i in {0..9}; do
    python main.py --run inference --local --model diabetica-7B --max_tokens 1500 --prompt_template 001 --do_sample --temperature 0.3
done