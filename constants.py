from os.path import dirname, join, realpath

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
TRAIN, DEV, TEST = 'train', 'dev', 'test'

# Model types (English)
EN_ENTITY_MODEL = 'EN_ENTITY_MODEL'
EN_EVENT_MODEL = 'EN_EVENT_MODEL'
EN_ENTITY_COREF_CONFIG = 'english_entity_coref'
EN_EVENT_COREF_CONFIG = 'english_event_coref'

# Model types (SPANISH)
ES_ENTITY_MODEL = 'ES_ENTITY_MODEL'
ES_EVENT_MODEL = 'ES_EVENT_MODEL'
ES_ENTITY_COREF_CONFIG = 'spanish_entity_coref'
ES_EVENT_COREF_CONFIG = 'spanish_event_coref'

MODEL_TYPES = [EN_ENTITY_MODEL, EN_EVENT_MODEL, ES_ENTITY_MODEL, ES_EVENT_MODEL]

# Pretrained model locations
PRETRAINED_EN_ENTITY_MODEL = '/shared/nas/data/m1/tuanml2/aida_entity_coref/pretrained/model.pt'
PRETRAINED_EN_EVENT_MODEL = '/shared/nas/data/m1/tuanml2/aida_coref_models/trained_english/model.pt'
PRETRAINED_ES_ENTITY_MODEL = '/shared/nas/data/m1/tuanml2/aida_entity_coref/spanish/model.pt'
PRETRAINED_ES_EVENT_MODEL = '/shared/nas/data/m1/tuanml2/aida_coref_models/trained_spanish/model.pt'
