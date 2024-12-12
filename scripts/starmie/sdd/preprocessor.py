import pandas as pd
import math
import collections
from collections import OrderedDict
import string
from pandas.api.types import infer_dtype

def computeTfIdf(tableDf):
    """ Compute tfIdf of each column independently """
    def computeTf(wordDict, doc):
        tfDict = {}
        docCount = len(doc)
        for word, count in wordDict.items():
            tfDict[word] = count / float(docCount)
        return tfDict

    def computeIdf(docList):
        idfDict = dict.fromkeys(docList.keys(), 0)
        N = len(docList)
        for word, val in docList.items():
            if val > 0:
                idfDict[word] += 1
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))
        return idfDict

    idf = {}
    for column in tableDf.columns:
        colVals = [val for entity in tableDf[column] for val in str(entity).split(' ')]
        wordSet = set(colVals)
        wordDict = dict.fromkeys(wordSet, 0)
        for val in colVals:
            wordDict[str(val)] += 1
        idf.update(computeIdf(wordDict))
    return idf

def pmiSample(val_counts, table, colIdxs, currIdx, max_tokens):
    """ PMI sampling (unchanged) """
    tokens = []
    valPairs = []
    topicCol = table[colIdxs[0]]
    PMIs = {}
    for i in range(topicCol.shape[0]):
        topicVal = topicCol[i]
        propVal = table.at[i, currIdx]
        if (topicVal, propVal) in val_counts and topicVal in val_counts and propVal in val_counts:
            pair_pmi = val_counts[(topicVal, propVal)] / (val_counts[topicVal] * val_counts[propVal])
            PMIs[(topicVal, propVal)] = pair_pmi
    PMIs = {k: v for k, v in sorted(PMIs.items(), key=lambda item: item[1], reverse=True)}
    if colIdxs.index(currIdx) == 0:
        valPairs = [k[0] for k in PMIs.keys()]
    else:
        valPairs = [k[1] for k in PMIs.keys()]
    for val in valPairs:
        for v in str(val).split(' '):
            if v not in tokens:
                tokens.append(v)
        if len(tokens) >= max_tokens:
            break
    return tokens

def constantSample(colVals, max_tokens):
    '''Constant sampling: take nth elements'''
    step = math.ceil(len(colVals) / max_tokens)
    tokens = colVals[::step]
    while len(tokens) > max_tokens:
        step += 1
        tokens = colVals[::step]
    return tokens

def frequentSample(colVals, max_tokens):
    '''Frequent sampling: most frequent tokens'''
    tokens = []
    tokenFreq = collections.Counter(colVals)
    # Break ties by lex order for stability
    tokenFreq_items = sorted(tokenFreq.items(), key=lambda x: (-x[1], x[0]))
    tokenFreq = dict(tokenFreq_items[:max_tokens])
    for t in colVals:
        if t in tokenFreq and t not in tokens:
            tokens.append(t)
    return tokens

def tfidfSample(column, tfidfDict, method, max_tokens):
    '''TF-IDF sampling: stable ordering'''
    tokens, tokenList, tokenFreq = [], [], {}

    # Sort unique values lexicographically to ensure invariance
    unique_values = sorted(column.astype(str).unique(), key=lambda x: str(x))

    if method == "tfidf_token":
        # token level
        for colVal in unique_values:
            for val in str(colVal).split(' '):
                idf = tfidfDict[val]
                tokenFreq[val] = idf
                tokenList.append(val)
        
        # Sort by IDF desc, then by token lex order for tie-breaking
        tokenFreq_items = sorted(tokenFreq.items(), key=lambda item: (-item[1], item[0]))
        tokenFreq = dict(tokenFreq_items[:max_tokens])

        for t in tokenList:
            if t in tokenFreq and t not in tokens:
                tokens.append(t)
                
    elif method == "tfidf_entity":
        # entity level
        entityScores = {}
        for colVal in unique_values:
            valIdfs = []
            vals = str(colVal).split(' ')
            for val in vals:
                valIdfs.append(tfidfDict[val])
            idf = sum(valIdfs)/len(valIdfs)
            entityScores[colVal] = idf
            tokenList.append(colVal)

        # Sort entityScores by IDF desc, then by entity string for tie-breaking
        entityScores_items = sorted(entityScores.items(), key=lambda item: (-item[1], str(item[0])))
        entityScores = OrderedDict(entityScores_items)

        # determine how many entities we can take to not exceed max_tokens
        valCount, N = 0, 0
        for entity, score in entityScores.items():
            valCount += len(str(entity).split(' '))
            if valCount < max_tokens:
                N += 1
            else:
                break
        # Keep top N entities
        chosen_entities = list(entityScores.keys())[:N]
        for t in tokenList:
            if t in chosen_entities and t not in tokens:
                tokens += str(t).split(' ')
    return tokens

def tfidfRowSample(table, tfidfDict, max_tokens):
    '''TF-IDF row sampling: ensure invariance by sorting rows first.'''
    # Sort table rows by all columns to ensure stable ordering
    # even if original row order is changed.
    table = table.copy()
    # Convert all columns to string for sorting consistency
    for c in table.columns:
        table[c] = table[c].astype(str)
    table = table.sort_values(by=list(table.columns))

    tokenFreq = {}
    for row in table.itertuples():
        index = row.Index
        rowVals = [val for entity in list(row[1:]) for val in str(entity).split(' ')]
        valIdfs = [tfidfDict[val] for val in rowVals]
        idf = sum(valIdfs)/len(valIdfs)
        tokenFreq[index] = idf

    # Sort rows by IDF desc, then by index to break ties
    tokenFreq_items = sorted(tokenFreq.items(), key=lambda item: (-item[1], item[0]))
    sortedRowInds = [k for k, v in tokenFreq_items[:max_tokens]]

    table = table.reindex(sortedRowInds)
    return table

def preprocess(column: pd.Series, tfidfDict: dict, max_tokens: int, method: str): 
    '''Preprocess a column into a list of max_tokens number of tokens.'''
    tokens = []
    colVals = [val for entity in column for val in str(entity).split(' ')]
    if method == "head" or method == "tfidf_row":
        # head: take first unique tokens in order of appearance
        # To ensure invariance, we sort unique values lexicographically
        unique_sorted = sorted(set(colVals), key=lambda x: str(x))
        for val in unique_sorted:
            if val not in tokens:
                tokens.append(val)
                if len(tokens) >= max_tokens:
                    break
    elif method == "alphaHead":
        # Sort entire column lexicographically and select top tokens
        if 'mixed' in infer_dtype(column):
            column = column.astype(str)
        sortedCol = column.astype(str).sort_values()
        sortedColVals = [val.lower() for entity in sortedCol for val in str(entity).split(' ')]
        # Unique sorted tokens are already stable
        seen = set()
        for val in sortedColVals:
            if val not in seen:
                seen.add(val)
                tokens.append(val)
                if len(tokens) >= max_tokens:
                    break
    elif method == "random":
        # random is by definition not stable, but if we want invariance,
        # we can't use randomness. If invariance is required, remove randomness or use a fixed seed.
        # Assuming invariance is not required for random:
        tokens = pd.Series(colVals).sample(min(len(colVals), max_tokens), random_state=0).sort_index().tolist()
    elif method == "constant":
        tokens = constantSample(colVals, max_tokens)
    elif method == "frequent":
        tokens = frequentSample(colVals, max_tokens) 
    elif "tfidf" in method and method != "tfidf_row":
        tokens = tfidfSample(column, tfidfDict, method, max_tokens)
    return tokens