# Round 1
python generate_question.py --file ./data/seed.json --save_path generation/complicate_0_cot.json --ins_file instruction/complicate.txt
python complicate_filter.py --file generation/complicate_0_cot.json --save_path generation/complicate_filtered_0_cot.json
python generate_cot.py --file generation/complicate_filtered_0_cot.json --save_path generation/complicate_1.json
python filter_chatgpt.py --file generation/complicate_1.json --save_path generation/complicate_chatgpt_filtered_1.json
python filter_claude.py --file generation/complicate_1.json --save_path generation/complicate_claude_filtered_1.json
python filter_palm.py --file generation/complicate_1.json --save_path generation/complicate_palm_filtered_1.json
python filter.py --chatgpt generation/complicate_chatgpt_filtered_1.json --claude generation/complicate_claude_filtered_1.json --palm generation/complicate_palm_filtered_1.json --save_path generation/filtered_1.json
# Round 2
python generate_question.py --file generation/filtered_1.json --save_path generation/complicate_1_cot.json --ins_file instruction/complicate.txt
python complicate_filter.py --file generation/complicate_1_cot.json --save_path generation/complicate_filtered_1_cot.json
python generate_cot.py --file generation/complicate_filtered_1_cot.json --save_path generation/complicate_2.json
python filter_chatgpt.py --file generation/complicate_2.json --save_path generation/complicate_chatgpt_filtered_2.json
python filter_claude.py --file generation/complicate_2.json --save_path generation/complicate_claude_filtered_2.json
python filter_palm.py --file generation/complicate_2.json --save_path generation/complicate_palm_filtered_2.json
python filter.py --chatgpt generation/complicate_chatgpt_filtered_2.json --claude generation/complicate_claude_filtered_2.json --palm generation/complicate_palm_filtered_2.json --save_path generation/filtered_2.json
# Round 3
python generate_question.py --file generation/filtered_2.json --save_path generation/complicate_2_cot.json --ins_file instruction/complicate.txt
python complicate_filter.py --file generation/complicate_2_cot.json --save_path generation/complicate_filtered_2_cot.json
python generate_cot.py --file generation/complicate_filtered_2_cot.json --save_path generation/complicate_3.json
python filter_chatgpt.py --file generation/complicate_3.json --save_path generation/complicate_chatgpt_filtered_3.json
python filter_claude.py --file generation/complicate_3.json --save_path generation/complicate_claude_filtered_3.json
python filter_palm.py --file generation/complicate_3.json --save_path generation/complicate_palm_filtered_3.json
python filter.py --chatgpt generation/complicate_chatgpt_filtered_3.json --claude generation/complicate_claude_filtered_3.json --palm generation/complicate_palm_filtered_3.json --save_path generation/filtered_3.json
# Round 4
python generate_question.py --file .generation/filtered_3.json --save_path generation/complicate_3_cot.json --ins_file instruction/complicate.txt
python complicate_filter.py --file generation/complicate_3_cot.json --save_path generation/complicate_filtered_3_cot.json
python generate_cot.py --file generation/complicate_filtered_3_cot.json --save_path generation/complicate_4.json
python filter_chatgpt.py --file generation/complicate_4.json --save_path generation/complicate_chatgpt_filtered_4.json
python filter_claude.py --file generation/complicate_4.json --save_path generation/complicate_claude_filtered_4.json
python filter_palm.py --file generation/complicate_4.json --save_path generation/complicate_palm_filtered_4.json
python filter.py --chatgpt generation/complicate_chatgpt_filtered_4.json --claude generation/complicate_claude_filtered_4.json --palm generation/complicate_palm_filtered_4.json --save_path generation/filtered_4.json