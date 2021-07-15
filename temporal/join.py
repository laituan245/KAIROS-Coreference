import json

data = json.load(open('temporal_input.json', 'r'))
data['temporal'] = json.load(open('temporal_response.txt', 'r'))
with open('final.json', 'w+') as f:
    f.write(json.dumps(data))

with open('temporal_relation.cs', 'w+') as f:
    f.write(data['temporal']['en']['temporal_relation.cs'])
