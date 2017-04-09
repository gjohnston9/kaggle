import pandas as pd
import numpy as np
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet, wordnet_ic
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp

import code
from collections import OrderedDict
import itertools
import re

import time

### http://staffwww.dcs.shef.ac.uk/people/S.Fernando/pubs/clukPaper.pdf

INFO_CONTENT_FILENAME = "ic-brown.dat" ### TODO: try other files, like ic-semcor.dat
ic_dict = wordnet_ic.ic(INFO_CONTENT_FILENAME)

df = pd.read_csv("train.csv")
print("number of rows: {}".format(len(df)))
sample = df.head(n=400)
all_questions = pd.concat([sample["question1"], sample["question2"]]).reset_index(drop=True)
# all_questions = pd.concat([sample["question1"], sample["question2"]])
# vect = TfidfVectorizer(tokenizer=nltk.word_tokenize, preprocessor = lambda s: re.sub(r'[^\w\s]', '', s, re.UNICODE).lower())
# X = vect.fit_transform(all_questions)

# for word, score in sorted(zip(vect.get_feature_names(), vect.idf_), key=lambda x: x[1]):
# 	print("{}: {}".format(word, score))

# totals = {0:0, 1:0}
# for i, dup in enumerate(sample.is_duplicate):
# 	q1 = X[2*i]
# 	q2 = X[2*i + 1]
# 	totals[dup] += q1.A.T.flatten().dot(q2.A.flatten())

# for dup in (0, 1):
# 	totals[dup] /= sample.is_duplicate.value_counts()[dup]

# print(totals)

stopwords = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','is','of','while','during','to']

def get_similarity(synsets1, synsets2, measure, ic_required):
	### - takes in two lists of synsets (with each list referring to a particular word,	and each synset in that list 
	### 	referring to a particular meaning of the word)
	### - estimates the similarity between the two words by finding the pair of synsets with maximal similarity
	###		(according to whatever measure was passed in, i.e. jcn similarity) and returning that similarity score

	### only get similarity between synsets that refer to same part of speech
	### (nltk will raise an error otherwise)
	compatible_senses = filter(lambda (s1, s2): s1.pos() == s2.pos(),itertools.product(synsets1, synsets2))
	max_score = 0
	for sense1, sense2 in compatible_senses:
		try:
			if ic_required:
				score = measure(sense1, sense2, ic_dict)
			else:
				score = measure(sense1, sense2)
			max_score = max(max_score, score)
		except WordNetError:
			pass
	return min(1, max_score) ### change +INF results to 1


def preprocess_sentence(sent):
	### change to lowercase, remove punctuation, split, remove stopwords
	sent = sent.lower()
	sent = re.sub(r'[^\w\s]', '', sent, re.UNICODE)
	return filter(lambda word: word not in stopwords, sent.split())
	

print("building set of all words")
# all_words = set(itertools.chain(*map(lambda q: q.split(), all_questions)))
all_words = set(itertools.chain(*map(preprocess_sentence, all_questions)))
all_words = OrderedDict((word, i) for i, word in enumerate(all_words))

print("getting all synsets for all words")
all_synsets = {word: wordnet.synsets(word) for word in all_words}
# print(all_words)

# print("building similarity matrix")
# similarity_matrix = []
# for word1 in all_words:
# 	similarities = []
# 	for word2 in all_words:
# 		if word1 == word2:
# 			similarities.append(1) ### TODO: may need to use different value depending on similarity measure
# 			continue
# 		syns1 = all_synsets[word1]
# 		syns2 = all_synsets[word2]
# 		if not (syns1 and syns2): ### no synsets for at least one of these words
# 			### TODO: handling words with no synsets:
# 			### misspelled words are useless
# 			### i.e. some proper nouns (relating to current news) will have no synsets
# 			### words starting with an uppercase letter that have no synsets are definitely
# 			### more informative than words starting with a lowercase letter and having no synsets
# 			### ***note: wn.synset(word) seems to be case-insensitive
# 			similarities.append(0)
# 			continue
# 		sim = get_similarity(syns1, syns2, wordnet.jcn_similarity, True)
# 		similarities.append(sim)

# 	similarity_matrix.append(similarities)

### build sparse symmetric matrix
print("building sparse similarity matrix")
sparse_sim = sp.sparse.dok_matrix((len(all_words), len(all_words)), dtype=sp.float64)
for word1, i in all_words.iteritems():
	for word2, j in all_words.iteritems():
		if j > i:
			continue
		if word1 == word2:
			sparse_sim[i, j] = 1
			continue
		syns1 = all_synsets[word1]
		syns2 = all_synsets[word2]
		if not (syns1 and syns2):
			continue
		sim = get_similarity(syns1, syns2, wordnet.jcn_similarity, True)
		if sim != 0:
			sparse_sim[i, j] = sim
			sparse_sim[j, i] = sim


# for word, row in zip(all_words, similarity_matrix):
# 	print("{}: {}".format(["{:.1f}".format(item) for item in row], word))
# sim = np.asarray(similarity_matrix)

sim_csc = sparse_sim.tocsc()

def tokens_to_vector(tokens):
	vect = np.zeros(len(all_words))
	for token in tokens:
		vect[all_words[token]] = 1

def semantic_similarity(v1, v2):
	### get estimated similarity between sentences
	return sim_csc.dot(v1).dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compare(df_idx):
	q1 = df.loc[df_idx].question1
	q2 = df.loc[df_idx].question2
	q1v = tokens_to_vector(preprocess_sentence(q1))
	q2v = tokens_to_vector(preprocess_sentence(q2))
	return semantic_similarity(q1, q2)

code.interact(local=locals())
