import json, random
with open('captions_val2017.json') as f:
    data = json.load(f)
prompts = [x['caption'] for x in data['annotations'][:100]]
with open('bench_prompts.txt','w') as f:
    f.write('\n'.join(prompts))