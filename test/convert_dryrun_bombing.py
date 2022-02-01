import os
import json
from os.path import join

BASE_PATH = 'test/dryrun_bombing'
EN_BASE_PATH = join(BASE_PATH, 'en')
ES_BASE_PATH = join(BASE_PATH, 'es')
ALL_BASE_PATH = join(BASE_PATH, 'all')

# Create directories
def create_dir_if_not_exist(dir):
    if not os.path.exists(dir): os.makedirs(dir)
create_dir_if_not_exist(ALL_BASE_PATH)
create_dir_if_not_exist(EN_BASE_PATH)
create_dir_if_not_exist(ES_BASE_PATH)

# Read sample_response.txt
with open(join(BASE_PATH, 'sample_response.txt')) as f:
    data = json.loads(f.read())

# Write files (All)
with open(join(ALL_BASE_PATH, 'entity.cs'), 'w+') as f:
    f.write(data['entity.cs'])
with open(join(ALL_BASE_PATH, 'event.cs'), 'w+') as f:
    f.write(data['event.cs'])
with open(join(ALL_BASE_PATH, 'clusters.txt'), 'w+') as f:
    f.write(data['clusters.txt'])
with open(join(ALL_BASE_PATH, 'distractors.txt'), 'w+') as f:
    f.write(data['distrators.txt'])

# Write files (EN)
with open(join(EN_BASE_PATH, 'entity.cs'), 'w+') as f:
    f.write(data['en']['entity.cs'])
with open(join(EN_BASE_PATH, 'event.cs'), 'w+') as f:
    f.write(data['en']['event.cs'])

# Write files (ES)
with open(join(ES_BASE_PATH, 'entity.cs'), 'w+') as f:
    f.write(data['es']['entity.cs'])
with open(join(ES_BASE_PATH, 'event.cs'), 'w+') as f:
    f.write(data['es']['event.cs'])
