import http
import time
import json
import slack
import argparse
import warnings
from flask import Flask
from flask_slack import Slack

app = Flask(__name__)
slack_app = Slack(app)


def get_history():
    history = client.conversations_history(channel=channel_id)
    text = history['messages'][0]['text']
    before_text = history['messages'][1]['text']
    return text, before_text

def annotate(message):
    # send to defined channel
    client.chat_postMessage(channel=channel_id, text=message, as_user=True)
    # get response
    text, before_text_ = get_history()
    temp = ''
    while True:
        try:                                           
            temp, before_text = get_history()
        except http.client.IncompleteRead:
            temp = 'Typing'
            
        if "Please note" in temp:
            temp = before_text
        if temp != text and 'Typing' not in temp:
            break
        else:
            time.sleep(1)
    response = temp.replace('\n\n', '\n').strip()
    return response

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--userOAuthToken',default="xoxp-")
    parser.add_argument('--channel_id', default="")
    parser.add_argument('--dataset', default="cot")
    parser.add_argument('--file', default="")
    parser.add_argument('--save_path', type=str, help="path to save the result")
    args = parser.parse_args()
    
    channel_id = args.channel_id
    userOAuthToken = args.userOAuthToken
    client = slack.WebClient(token=userOAuthToken)
    dataset = args.dataset
    save_dir = args.save_path
    
    
    if dataset == 'cot':
        eval_data_path = args.file
        ins = "Given a question and an answer to the question, try your best to judge whether the answer is right or wrong. If it's right, write 'Yes'. If it's wrong, write 'No'.\n\n#Question#: "
        with open(eval_data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                each_data = json.loads(line)
                input = ins + each_data['question'] + "\n#Answer#: " + each_data['answer'] + "\n#Your Judgement#:"
                ans = ''
                while len(ans) == 0:
                    try:
                        ans = annotate(input)
                    except:
                        continue
                print(ans)
                ans = ans.split("/n")[0]
                if "Yes" in ans or "yes" in ans or "correct" in ans:
                    judgement = "Yes"
                elif "No" in ans or "no" in ans or "wrong" in ans:
                    judgement = "No"
                else:
                    judgement = "Unknown"
                dump_jsonl({"id": each_data['id'], "question": each_data["question"], "answer": each_data['answer'], "judgement": judgement}, save_dir)
                print(each_data["id"], "completed!")