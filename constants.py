from os.path import dirname, join, realpath

NOT_ENTITY = 'not_entity'

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
TRAIN, DEV, TEST = 'train', 'dev', 'test'

EVENT_MODEL = 'EVENT_MODEL'
EVENT_COREF_CONFIG = 'event_coref'

# Model types (English)
EN_ENTITY_MODEL = 'EN_ENTITY_MODEL'
EN_ENTITY_COREF_CONFIG = 'english_entity_coref'

# Model types (SPANISH)
ES_ENTITY_MODEL = 'ES_ENTITY_MODEL'
ES_ENTITY_COREF_CONFIG = 'spanish_entity_coref'

MODEL_TYPES = [EVENT_MODEL, EN_ENTITY_MODEL, ES_ENTITY_MODEL]

# Pretrained model locations
PRETRAINED_EVENT_MODEL = 'event.pt'
PRETRAINED_EN_ENTITY_MODEL = 'en_entity.pt'
PRETRAINED_ES_ENTITY_MODEL = 'es_entity.pt'
