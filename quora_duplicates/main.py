import pandas as pd
import numpy as np
import scipy as sp

from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet, wordnet_ic

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import code
from collections import OrderedDict
import itertools
import pdb
import re

### http://staffwww.dcs.shef.ac.uk/people/S.Fernando/pubs/clukPaper.pdf

INFO_CONTENT_FILENAME = "ic-brown.dat" ### TODO: try other files, like ic-semcor.dat
MAX_SIMILARITY = 500 ### similarity to use for words that are the same, and to bound the similarity of non-equal words

NUM_SAMPLES = 750 ### number of rows from dataframe to use

ic_dict = wordnet_ic.ic(INFO_CONTENT_FILENAME)

df = pd.read_csv("train.csv")
print("number of rows: {}".format(len(df)))
sample = df.head(n=NUM_SAMPLES)
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

stopwords = ['the', 'a', 'an', 'and', 'but', 'or', 'because','as','this','that','these','those','then',
              'just','so','than','such','both','through','about','for','of','while','during','to']

question_words = ['who', 'what', 'when', 'where', 'why', 'is', 'was', 'if', 'will', 'were']

assert(len(set(stopwords) & set(question_words)) == 0)

greater_than_one = []
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
	if max_score > 1:
		greater_than_one.append((max_score, synsets1, synsets2))
	return min(MAX_SIMILARITY, max_score) ### change +INF results to 500


def preprocess_sentence(sent):
	### change to lowercase, remove punctuation, split, remove stopwords
	sent = sent.lower()
	sent = re.sub(r'[^\w\s]', '', sent, re.UNICODE)
	return filter(lambda word: word not in stopwords, sent.split())
	

print("building set of all words")
all_words = set(itertools.chain(*map(preprocess_sentence, all_questions)))
all_words = OrderedDict((word, i) for i, word in enumerate(all_words))

print("getting all synsets for all words")
all_synsets = {word: wordnet.synsets(word) for word in all_words}

### build sparse symmetric matrix
print("building sparse similarity matrix")
sparse_sim = sp.sparse.dok_matrix((len(all_words), len(all_words)), dtype=sp.float64)
for word1, i in all_words.iteritems():
	for word2, j in all_words.iteritems():
		if j > i:
			continue
		if word1 == word2:
			sparse_sim[i, j] = MAX_SIMILARITY
			continue
		syns1 = all_synsets[word1]
		syns2 = all_synsets[word2]
		if not (syns1 and syns2):
			continue
		sim = get_similarity(syns1, syns2, wordnet.jcn_similarity, True)
		if sim != 0:
			sparse_sim[i, j] = sim
			sparse_sim[j, i] = sim

sim_csc = sparse_sim.tocsc()
### can't directly use np.where on sparse matrices. There are more efficient approaches than this but it doesn't
### seem to take too long anyway
dense = sim_csc.todense()

truncated = np.where(dense >= 0.8, dense, 0) ### change all values < 0.8 to 0
truncated_csc = sp.sparse.csc_matrix(truncated)

def tokens_to_vector(tokens):
	vect = np.zeros(len(all_words))
	for token in tokens:
		vect[all_words[token]] = 1
	return vect

def semantic_similarity(v1, v2, semantic_matrix=truncated_csc):
	### get estimated similarity between sentences
	return semantic_matrix.dot(v1).dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def build_data(limit):
	X = []
	y = []
	for index, row in df[:limit].iterrows():
		q1 = preprocess_sentence(row.question1)
		q2 = preprocess_sentence(row.question2)
		q1rev = q1[::-1]
		q2rev = q2[::-1]
		q1v = tokens_to_vector(q1)
		q2v = tokens_to_vector(q2)
		fwd_differences = filter(lambda i: q1[i] != q2[i], xrange(min(len(q1), len(q2)))) + [0]
		rev_differences = filter(lambda i: q1rev[i] != q2rev[i], xrange(min(len(q1), len(q2)))) + [0]

		example_similarity = semantic_similarity(q1v, q2v)
		question_word_overlap = sum(1 for word in question_words if (word in q1) and (word in q2))
		first_difference = fwd_differences[0] ### position of first character that differs between q1 and q2
		first_rev_difference = rev_differences[0] ### position of first character that differs between q1[::-1] and q2[::-1]

		X.append([example_similarity, question_word_overlap, first_difference, first_rev_difference])
		y.append(row.is_duplicate)

	return X, y

def build_classifier(test_size=0.3):
	X, y = build_data(NUM_SAMPLES)
	clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5,2,)) ### TODO: experiment with parameters...
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	clf.fit(X_train, y_train)
	clf.score(X_test, y_test)
	pdb.set_trace()

code.interact(local=locals())
