from os.path import dirname, join, realpath

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
ENTITY_COREF_CONFIG = 'entity_coref'
EVENT_COREF_CONFIG = 'event_coref'
TRAIN, DEV, TEST = 'train', 'dev', 'test'

# Model types
ENTITY_MODEL = 'ENTITY'
EVENT_MODEL = 'EVENT'
MODEL_TYPES = [ENTITY_MODEL, EVENT_MODEL]

# Pretrained model locations
PRETRAINED_ENTITY_MODEL = '/shared/nas/data/m1/tuanml2/aida_entity_coref/pretrained/model.pt'
PRETRAINED_EVENT_MODEL = '/shared/nas/data/m1/tuanml2/aida_coref_models/trained_english/model.pt'
