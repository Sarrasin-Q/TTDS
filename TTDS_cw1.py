import argparse
import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict


"""
     As the code is run locally on my Pycharm, the folders where the files are read and stored are on the local computer,
and need to be changed if the you want to re-run the code in your computer.
"""


# Pre-processes text
def process_text(text):
    """Pre-processing of text, including tokenisation, case-folding, stopping and Porter stemming
    Args:
        text (str): text to be processed, containing more than one word
    Returns:
        tokens (list): a list of tokens obtained after processing, each of which is used for indexing and searching
    """
    # Tokenisation
    tokens = re.findall(r'\w+', text)
    # Case folding
    tokens_lower = [word.lower() for word in tokens]
    # Stopping & Porter stemming
    porter_stemmer = PorterStemmer()
    tokens = [porter_stemmer.stem(t.lower()) for t in tokens_lower if not t in ST]
    return tokens

# Pre-processes query word
def process_word(word):
    """Pre-processing of a query word, including tokenisation, case-folding, stopping and Porter stemming
       Args:
           word (str): single word to be processed
       Returns:
           token (str): a token obtained after processing, which is used as a search index
       """
    word = word.lower()
    if word not in ST:
        porter_stemmer = PorterStemmer()
        word = porter_stemmer.stem(word)
        return word
    return ''



# Load the documents collection
def load_xml(file):
    """Parsing xml files and storing all document information, including Docid and content (headline & text)
    Args:
        file: xml file collection of all documents
    Returns:
        docs_df (dataframe): docs_df.loc[docID] = content
    """
    docs_df = pd.DataFrame(columns=['doc'])
    for doc in ET.fromstring(open(file, 'r', encoding='utf-8').read()):
        docID = doc.find('DOCNO').text
        # Remember to include the headline and text to content
        content = doc.find('HEADLINE').text + doc.find('TEXT').text
        docs_df.loc[docID] = content
    return docs_df

# Creates a positional inverted index
def creat_index(docs_df):
    """Create a positional inverted index based on document information
    Args:
        docs_df (dataframe): storage of all document information
    Returns:
        index (OrderedDict): the positional inverted index
        (# index[token] = Docs[ ] = posting of the token = document IDs where the token appeared)
        (# Docs[docID] = lists of positions of the token in each relevant document)
    """
    index = defaultdict(lambda: defaultdict(list))
    for i, dp in docs_df.iterrows():
        doc = dp['doc']
        # Preprocess the text and get tokens for indexing
        tokens = process_text(doc)
        for pos in range(len(tokens)):
            index[tokens[pos]][int(i)] += [pos + 1]
    index = OrderedDict(sorted(index.items()))
    return index

# Write index to the file
def write_index_file(index):
    """Write the positional inverted index we created to index.txt
    Args:
        index (OrderedDict): the positional inverted index
    """
    with open('C://Users//22398//Desktop//TTDS//cw1//index.txt', 'w+', encoding='utf-8') as f:
        for word, docs in index.items():
            # Calculate the document frequency of the token
            df = len(docs)
            indexing = word + ':' + str(df) + '\n'
            for doc, pos in docs.items():
                indexing += '\t' + str(doc) + ':' + ' '
                for each in pos:
                    if each == pos[-1]:
                        indexing += str(each)
                    else:
                        indexing += str(each) + ', '
                indexing += '\n'
            f.write(indexing + '\n')
    f.close()



# Phrase search
def phrase_search(phrase):
    """
    Args: Implementing phrase search
        phrase(str): phrase query with the structure "word1, word2"
    Returns:
        result (list): A list of the document ids corresponding to the phrase query
    """
    result = []
    # Get two words for the query phrase
    phrase = phrase.strip('"').split(' ')
    # Remember to proprecess the word before searching
    word1 = process_word(phrase[0])
    word2 = process_word(phrase[-1])
    # Make sure the two query terms are in the same article and in consecutive
    if word1 in index.keys() and word2 in index.keys():
        docs1 = index.get(word1)
        docs2 = index.get(word2)
        for doc1 in docs1.keys():
            if doc1 in docs2.keys():
                value1 = docs1[doc1]
                value2 = docs2[doc1]
                for pos in value1:
                    if pos+1 in value2:
                        result.append(int(doc1))
                        break
    return result

# Proximity search
def proximity_search(proximity):
    """
    Args: Implementing proximity search
        phrase(str): proximity query with the structure #num(word1, word2)
    Returns:
        result (list): A list of the document ids corresponding to the proximity query
    """
    result = []
    # Get the two words in the query and the maximum distance requested
    num = proximity.split('#')[1].split('(')[0].strip(' ')
    word1 = proximity.split('(')[1].split(',')[0].strip(' ')
    word2 = proximity.split(',')[1].split(')')[0].strip(' ')
    word1 = process_word(word1)
    word2 = process_word(word2)
    # Make sure that two tokens are in the same document and no more than the required interval between them
    if word1 in index.keys() and word2 in index.keys():
        docs1 = index.get(word1)
        docs2 = index.get(word2)
        for doc1 in docs1.keys():
            if doc1 in docs2.keys():
                value1 = docs1[doc1]
                value2 = docs2[doc1]
                for pos in value1:
                    for i in range(1, int(num)+1):
                        # The order is not important
                        if pos + i in value2 or pos - i in value2:
                            result.append(int(doc1))
                            break
                    else:
                        continue
                    break
    return result

# Search a single word
def single_search(word):
    """
    Args: Implementing single word search
        word(str): only one word query
    Returns:
        result (list): A list of the document ids corresponding to the single word query
    """
    result = []
    word = process_word(word)
    if word in index.keys():
        for doc in index[word].keys():
            result.append(int(doc))
    return result

def search(word):
    """Get search results for every query term
    Args:
        word(str): query term (can be with NOT), of word/ phrase/ proximity structure
    Returns:
        result(list): A list of the document ids corresponding to the query term
    """
    # Define a bool value haveNot to check is there a NOT in query term
    haveNot = False
    alldocs = []
    for i in range(0, docamount):
        alldocs.append(int(docs_df.index.values[i]))
    if word.find('NOT') == 0:
        # If there is a NOT, define havenNot as true and remove the NOT for subsequent search
        haveNot = True
        word = word.split(' ')[-1]
    if word.startswith('"'):
        # Phrase search
        result = phrase_search(word)
    elif word.startswith('#'):
        # Proximity search
        result = proximity_search(word)
    else:
        # Single search
        result = single_search(word)

    if len(result) > 0:
        if haveNot:
            # If there is a NOT, the final result obtained is the full set minus the searched dataset
            return list(set(alldocs) - set(result))
        else:
            return result

# Implement boolean search
def boolean_search():
    """Implementation of  Boolean search, Phrase search, Proximity search with logical operator
    Returns:
        For each query, write the Docid of the corresponding documents to results.boolean.txt
    """
    fread = open('C://Users//22398//Desktop//TTDS//cw1//queries.boolean.txt',
                 'r', encoding='utf-8')
    fwrite = open('C://Users//22398//Desktop//TTDS//cw1//results.boolean.txt',
                  'w', encoding='utf-8')

    for query in fread:
        querywords = []

        query = re.split(r' +(AND|OR) +', query)
        num = query[0].split(' ', 1)[0].strip()
        word1 = query[0].split(' ', 1)[1].strip()

        querywords.append(word1)
        for item in query[1:]:
            querywords.append(item.strip())

        # Search with only one term
        if len(querywords) == 1:
            result = search(word1)
        else:
            # Get the two terms for the query
            word1 = querywords[0]
            word2 = querywords[-1]
            result1 = search(word1)
            result2 = search(word2)

            # Logical operator judgement
            if len(result1) > 0:
                if 'AND' in querywords:
                    # Get the intersection of the results
                    result = list(set(result1) & set(result2))
                if 'OR' in querywords:
                    # Get the resulting merge set
                    result = list(set(result1) | set(result2))

        result = sorted(result)
        for each in result:
            line = num + ', ' + str(each) + '\n'
            fwrite.write(line)

    fread.close()
    fwrite.close()



# Calculate TFIDF
def tfidf(querywords):
    """Calculates the retrieval score using the TFIDF (term frequency - inverse document frequency) formula
    Args:
        querywords(str): a query including several tokens
    Returns:
        sorted_result (list): sorted list according to the retrieval score of a query and a document
    """
    result = []
    idf = []
    subindex = []
    # Calculate the "idf" for each token in the query
    for term in querywords[1:]:
        if term in index.keys():
            docs = index[term]
            df = len(docs)
            # Store the "idf" values of all query tokens in a list idf[]
            idf.append(np.log10(int(docamount) / df))
            # Store the corresponding document ids that any query tokens appears, in a list subindex[]
            subindex.append(docs)
        else:
            idf.append(0)
            subindex.append({})
    # Read each document in turn
    for i in docs_df.index.values:
        w = 0
        # Process each token in turn
        for num in range(len(querywords) - 1):
        # If the token appears in the document, calculate its "tf"
            if int(i) in list(subindex[num].keys()):
                tf = len(subindex[num][int(i)])
                value = 1 + (np.log10(tf))
            else:
                value = 0
                # Calculate the w and sum them to get the score
            w = w + (value * idf[num])
        # Write the score of each query and each document to the list result[]
        result.append([int(querywords[0]), i, "{:.4f}".format(w)])
    # Sort the list result[]
    return sorted(result, key=lambda x: x[-1], reverse=-True)

# Implement ranked search
def ranked_search():
    """Implementation of Ranked IR
    Returns:
        For each query, write the Docid of the 150 most relevant documents to results.ranked.txt
    """
    fread = open('C://Users//22398//Desktop//TTDS//cw1//queries.ranked.txt',
                 'r', encoding='utf-8')
    fwrite = open('C://Users//22398//Desktop//TTDS//cw1//results.ranked.txt',
                  'w', encoding='utf-8')

    for query in fread:
        querywords = process_text(query)
        # For each query, calculate the score of that and each document based on TFIDF, and sort
        result = tfidf(querywords)
        # Output the 150 most relevant (i.e. highest score) documents
        for each in range(150):
            line = ''
            for item in result[each]:
                if item == result[each][-1]:
                    line += str(item)
                else:
                    line += str(item) + ', '
            line += '\n'
            fwrite.write(line)

    fread.close()
    fwrite.close()



if __name__ == '__main__':

    global ST
    global docs_df
    global index
    global docamount

    # Get the stopwords list
    ST = []
    with open('C://Users//22398//Desktop//TTDS//cw1//englishST.txt', 'r', encoding='utf-8') as f:
        for line in f:
            ST.append(line.replace('\n', ''))
    f.close()

    # Load the xml file
    docs_df = load_xml('C://Users//22398//Desktop//TTDS//cw1//trec.5000.xml')
    docamount = len(docs_df)
    # Creat the positional inverted index
    index = creat_index(docs_df)
    # Write the index to index.txt
    write_index_file(index)
    # Get the results of given queries and write them to results.boolean.txt
    boolean_search()
    # Get the results of given queries and write them to results.ranked.txt
    ranked_search()



