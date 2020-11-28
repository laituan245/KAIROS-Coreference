from scripts import es_translation
en2es = es_translation('resources/quizlet4/es/linking/', 'resources/quizlet4/en/linking/')

with open('en_to_es.txt', 'w+') as f:
    for en in en2es:
        f.write('{} -> {}\n'.format(en, en2es[en]))
