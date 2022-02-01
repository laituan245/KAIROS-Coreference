import json
from elasticsearch import Elasticsearch

# Elasticsearch Configs
ES_HOST = 'wikidata-es'
EN_WIKIDATA_INDEX = 'en_wikidata'

# Helper Functions
def multi_match_clause(query_text, fields):
    body = {
        'multi_match':{
            'query': query_text,
            'fields': fields,
            'fuzziness' : 'AUTO'
        }
    }
    return body

def term_clause(query_text, field, boost_score=2.0):
    body = {
        'term': {
            f'{field}.raw': {
                'value': query_text,
                'boost': boost_score
            }
        }
    }
    return body

def get_result(r):
    data = r['_source']
    data['_id'] = r['_id']
    return data

# Main Class
class ESCandidateRetriever:
    def __init__(self, topk=10):
        self.es = Elasticsearch([ES_HOST], timeout=30)
        self.topk = topk

    def search_entities_by_ids(self, entity_ids):
        search_results = self.es.search(
            index = EN_WIKIDATA_INDEX,
            body = {
                'query': {
                    'ids': {'values': entity_ids}
                },
                'size': len(entity_ids)
            }
        )
        search_results = [get_result(r) for r in search_results['hits']['hits']]
        return search_results

    def msearch_candidates(self, titles, structured_queries=None, fields=None):
        search_arr, batch_size = [], len(titles)
        for ix, title in enumerate(titles):
            # req_head
            search_arr.append({'index': EN_WIKIDATA_INDEX})
            # req_body
            req_body = {
                'query': {
                  'bool': {
                    'should': [
                        multi_match_clause(title, ['title_and_aliases']),
                        term_clause(title, 'title_and_aliases'),
                    ]
                  }
                }
            }
            # Structured queries (if any)
            if structured_queries:
                if structured_queries[ix]:
                    es = structured_queries[ix].split('|')
                    es = [e.strip() for e in es]
                    req_body['query']['bool']['should'].append(multi_match_clause(es[0], ['title_and_aliases']))
                    req_body['query']['bool']['should'].append(term_clause(es[0], ['title_and_aliases']))
                    if len(es) > 1:
                        req_body['query']['bool']['should'].append(multi_match_clause(es[1], ['description']))
            # Other fields
            if not fields is None: req_body['_source'] = fields
            req_body['from'] = 0
            req_body['size'] = self.topk
            search_arr.append(req_body)

        request = ''
        for each in search_arr:
            request += '%s \n' %json.dumps(each)

        # as you can see, you just need to feed the <body> parameter,
        # and don't need to specify the <index> and <doc_type> as usual
        resp = self.es.msearch(body = request)['responses']
        assert(len(resp) == len(titles))

        # Finalize mseach_results
        mseach_results = []
        for i in range(batch_size):
            search_results = resp[i]
            search_results = [get_result(r) for r in search_results['hits']['hits']]
            mseach_results.append(search_results)
        return mseach_results
