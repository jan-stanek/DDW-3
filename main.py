import nltk
import wikipedia
import csv
from collections import Counter


text = None
with open('data.txt', 'r') as f:
    text = f.read()

tokens = nltk.word_tokenize(text)

def tokenCounts(tokens):
    counts = Counter(tokens)
    sortedCounts = sorted(counts.items(), key=lambda count: count[1], reverse=True)
    return sortedCounts

def wiki(entity):
    results = wikipedia.search(entity)
    if len(results) == 0:
        return [entity, 'Thing']

    try:
        page = wikipedia.page(title=results[0], auto_suggest=False)
        sent = nltk.sent_tokenize(page.summary)[0]
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)

        isPos = -1
        for i in range(len(tagged)):
            if tagged[i][0] == 'is' or tagged[i][0] == 'are':
                isPos = i
                break
        if isPos == -1:
            return [entity, 'Thing']

        nnFound = False
        res = []

        for i in range(isPos+1, len(tagged)):
            if nnFound and tagged[i][1] != 'NN':
                break
            if tagged[i][1] == 'NN':
                nnFound = True
            res.append(tagged[i][0])

        return [entity, ' '.join(res)]
    except:
        return [entity, 'Thing']


# POS
tagged = nltk.pos_tag(tokens)
print(tokenCounts(tagged))

# NER - NLTK
def extractEntities(ne_chunked):
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
        else:
            continue
    return data

ne_chunked = nltk.ne_chunk(tagged, binary=False)
ne = [token for token in extractEntities(ne_chunked)]
ne_types = extractEntities(ne_chunked)

with open('nltk.csv', 'w') as csvfile:
    fieldnames = ['entity', 'classification']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for r in ne:
        writer.writerow({'entity': r, 'classification': ne_types[r]})

res = []
for e in ne:
    res.append(wiki(e))

with open('nltk-wiki.csv', 'w') as csvfile:
    fieldnames = ['entity', 'classification']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for r in res:
        writer.writerow({'entity': r[0], 'classification': r[1]})



# NER - custom
grammar = "NP: {<DT>?<JJ>*<NN|NNS>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(tagged)
ne = []
for res in result:
    if isinstance(res, nltk.tree.Tree) and (res[0][1] == 'NNS' or res[0][1] == 'NN'):
        ne.append(res[0][0])
res = []
for e in ne:
    res.append(wiki(e))
with open('custom-wiki.csv', 'w') as csvfile:
    fieldnames = ['entity', 'classification']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    for r in res:
        writer.writerow({'entity': r[0], 'classification': r[1]})






