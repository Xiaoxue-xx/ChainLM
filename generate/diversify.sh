# Round 1
python generate_question.py --file ./data/seed.json --save_path generation/diversify_0_cot.json --ins_file instruction/diversify.txt
python diversify_filter.py --file generation/diversify_0_cot.json --save_path generation/diversify_filtered_0_cot.json
python generate_cot.py --file generation/diversify_filtered_0_cot.json --save_path generation/diversify_1.json
python filter_chatgpt.py --file generation/diversify_1.json --save_path generation/diversify_chatgpt_filtered_1.json
python filter_claude.py --file generation/diversify_1.json --save_path generation/diversify_claude_filtered_1.json
python filter_palm.py --file generation/diversify_1.json --save_path generation/diversify_palm_filtered_1.json
python filter.py --chatgpt generation/diversify_chatgpt_filtered_1.json --claude generation/diversify_claude_filtered_1.json --palm generation/diversify_palm_filtered_1.json --save_path generation/filtered_1.json
# Round 2
python generate_question.py --file generation/filtered_1.json --save_path generation/diversify_1_cot.json --ins_file instruction/diversify.txt
python diversify_filter.py --file generation/diversify_1_cot.json --save_path generation/diversify_filtered_1_cot.json
python generate_cot.py --file generation/diversify_filtered_1_cot.json --save_path generation/diversify_2.json
python filter_chatgpt.py --file generation/diversify_2.json --save_path generation/diversify_chatgpt_filtered_2.json
python filter_claude.py --file generation/diversify_2.json --save_path generation/diversify_claude_filtered_2.json
python filter_palm.py --file generation/diversify_2.json --save_path generation/diversify_palm_filtered_2.json
python filter.py --chatgpt generation/diversify_chatgpt_filtered_2.json --claude generation/diversify_claude_filtered_2.json --palm generation/diversify_palm_filtered_2.json --save_path generation/filtered_2.json
# Round 3
python generate_question.py --file generation/filtered_2.json --save_path generation/diversify_2_cot.json --ins_file instruction/diversify.txt
python diversify_filter.py --file generation/diversify_2_cot.json --save_path generation/diversify_filtered_2_cot.json
python generate_cot.py --file generation/diversify_filtered_2_cot.json --save_path generation/diversify_3.json
python filter_chatgpt.py --file generation/diversify_3.json --save_path generation/diversify_chatgpt_filtered_3.json
python filter_claude.py --file generation/diversify_3.json --save_path generation/diversify_claude_filtered_3.json
python filter_palm.py --file generation/diversify_3.json --save_path generation/diversify_palm_filtered_3.json
python filter.py --chatgpt generation/diversify_chatgpt_filtered_3.json --claude generation/diversify_claude_filtered_3.json --palm generation/diversify_palm_filtered_3.json --save_path generation/filtered_3.json
# Round 4
python generate_question.py --file .generation/filtered_3.json --save_path generation/diversify_3_cot.json --ins_file instruction/diversify.txt
python diversify_filter.py --file generation/diversify_3_cot.json --save_path generation/diversify_filtered_3_cot.json
python generate_cot.py --file generation/diversify_filtered_3_cot.json --save_path generation/diversify_4.json
python filter_chatgpt.py --file generation/diversify_4.json --save_path generation/diversify_chatgpt_filtered_4.json
python filter_claude.py --file generation/diversify_4.json --save_path generation/diversify_claude_filtered_4.json
python filter_palm.py --file generation/diversify_4.json --save_path generation/diversify_palm_filtered_4.json
python filter.py --chatgpt generation/diversify_chatgpt_filtered_4.json --claude generation/diversify_claude_filtered_4.json --palm generation/diversify_palm_filtered_4.json --save_path generation/filtered_4.json