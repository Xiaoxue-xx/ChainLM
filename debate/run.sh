python debate4cot.py --model-path {model-path} --test-file test_data/commonsenceqa/test_new.json --answer-file csqa.json --prompt instruction/csqa.txt --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/siqa/test_new.json --answer-file siqa.json --prompt instruction/siqa.txt --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/math/elementary_mathematics.json --answer-file elemath.json --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/math/test_new.json --answer-file math.json --prompt instruction/math.txt --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/scienceqa/test_new.json --answer-file scienceqa.json --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/SciQ/test.json --answer-file sciq.json --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/penguins_in_a_table/valid.json --answer-file penguins.json --device cuda:0
python debate4cot.py --model-path {model-path} --test-file test_data/o_counting/valid.json --answer-file counting.json --prompt instruction/counting.txt --device cuda:0