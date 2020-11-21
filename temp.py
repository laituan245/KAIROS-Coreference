from scripts.docs_clustering import docs_clustering

EN_DISTRACTORS = set(['K0C0448WM', 'K0C0448WL', 'K0C0448WJ', 'K0C0448WI'])
EN_DOCS_1 = set(['K0C047Z59', 'K0C047Z57', 'K0C041NHY', 'K0C041NHW', 'K0C041NHV'])
EN_DOCS_2 = set(['K0C047Z5A', 'K0C041O37', 'K0C041O3D'])

ALL_DISTRACTORS = EN_DISTRACTORS.union(EN_DOCS_1)
docs_clustering('resources/quizlet4/en/oneie/m1_m2/json', ALL_DISTRACTORS)
