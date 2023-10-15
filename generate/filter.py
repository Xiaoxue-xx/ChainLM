import argparse
import json


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chatgpt', type=str, default="")
    parser.add_argument('--claude', type=str, default="")
    parser.add_argument('--palm', type=str, default="")
    parser.add_argument('--save_path', type=str, default="")
    args = parser.parse_args()
    chatgpt = args.chatgpt
    claude = args.claude
    palm = args.palm

    chatgpt_data = []
    claude_data = []
    palm_data = []

    with open(chatgpt, 'r', encoding="utf-8") as f:
        for line in f:
            chatgpt_data.append(json.loads(line))
    with open(claude, 'r', encoding="utf-8") as f:
        for line in f:
            claude_data.append(json.loads(line))
    with open(palm, 'r', encoding="utf-8") as f:
        for line in f:
            palm_data.append(json.loads(line))

    for i in range(len(chatgpt_data)):
        if chatgpt_data[i]["judgement"] == "Yes" and claude_data[i]["judgement"] == "Yes" and palm_data[i]["judgement"] == "Yes":
            judgement = "Yes"
            dump_jsonl({"id":chatgpt_data[i]["id"], "question": chatgpt_data[i]["question"], "answer": chatgpt_data[i]["answer"]}, args.save_path, append=True)
        