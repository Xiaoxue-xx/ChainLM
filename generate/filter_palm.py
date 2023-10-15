import argparse
import google.generativeai as palm
import base64
import json

# Configure the client library by providing your API key.
palm.configure(api_key="")

def get_response(message):
  response = palm.chat(messages=message)
  return(response.last)

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, 'a+', encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

def get_dataset(args):
  with open(args.file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        i=0
        for i in range(len(data)):
            question = data[i]["question"]
            answer = data[i]["answer"]
            message = "Given a question and an answer to the question, try your best to judge whether the answer is right or wrong. If it's right, write 'Yes'. If it's wrong, write 'No'.\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"
            ans = get_response(message)
            if ans is None:
              judgement = "No"
            elif("\n" in ans):
              ans = ans.split("\n")[0]
              print(ans)
            if ans is None:
              judgement = "No"
            elif "correct" in ans or "yes" in ans or "Yes" in ans:
                judgement = "Yes"
            elif "wrong" in ans or "no" in ans or "No" in ans:
                judgement = "No"
            else:
              judgement = "No"
            gen = {"id":data[i]["id"],"question": question, "answer": answer, "judgement": judgement}
            dump_jsonl(gen, args.save_path, append=True)

            print(data[i]["id"], "completed!")
            i = i+1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    get_dataset(args)