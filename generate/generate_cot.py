import argparse
import openai
import time
import json

openai.api_key = 'sk-'

def get_res_batch(input):
    message = [
        {"role": "user", "content": "Answer the given question step by step. And put the answer after '####' in the end. Here is a example: \n\n#Question#: Does crucifixion violate US eighth amendment? Hmmm, my chain of thoughts:\n#Answer#: The eighth amendment prohibits cruel and unusual punishment. Crucifixion was particularly barbaric as people do not die instantly and live for several days. Final answer: yes.\n#### Yes\n\n#Question#: " + input + "\n#Answer#: "},
    ]
    while True:
        try:
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message,
                temperature=1.0,
                max_tokens=512
            )
            break
        except openai.error.RateLimitError:
            print('openai.error.RateLimitError\nRetrying...')
            time.sleep(60)
        except openai.error.ServiceUnavailableError:
            print('openai.error.ServiceUnavailableError\nRetrying...')
            time.sleep(20)
        except openai.error.Timeout:
            print('openai.error.Timeout\nRetrying...')
            time.sleep(20)
        except openai.error.APIError:
            print('openai.error.APIError\nRetrying...')
            time.sleep(20)
        except openai.error.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            time.sleep(20)
    
    
    print(res['choices'][0]['message']['content'])
    return res['choices'][0]['message']['content']

def get_dataset(args):
    with open(args.file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        i=0
        for i in range(len(data)):
            if data[i]["judgement"] == "No":
                continue
            input = data[i]["question"]

            ans = get_res_batch(input)

            if "id" not in data[i]:
                id = i
            else:
                id = data[i]["id"]
            gen = {"id":id,"question": input, "answer": ans}
            dump_jsonl(gen, args.save_path, append=True)
            i = i+1
            print(i, "completed!")

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
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()
    
    get_dataset(args)
