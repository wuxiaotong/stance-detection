import pandas
import gensim
import collections
import sklearn
import numpy as np
import math
from collections import defaultdict
from gensim import corpora, models, similarities
from sklearn.metrics.cluster import normalized_mutual_info_score

class UnigramLanguageModel:
	def __init__(self, sentences, smoothing=False):
		self.unigram_frequencies = dict()
		self.corpus_length = 0
		for sentence in sentences:
			for word in sentence:
				self.unigram_frequencies[word] = self.unigram_frequencies.get(word, 0) + 1
				# if word != SENTENCE_START and word != SENTENCE_END:
				self.corpus_length += 1
		self.unique_words = len(self.unigram_frequencies) - 2
		self.smoothing = smoothing

	def calculate_unigram_probability(self, word):
		word_probability_numerator = self.unigram_frequencies.get(word, 0)
		word_probability_denominator = self.corpus_length
		if self.smoothing:
			word_probability_numerator += 1
			word_probability_denominator += self.unique_words + 1
		return float(word_probability_numerator) / float(word_probability_denominator)
	
	def calculate_sentence_probability(self, sentence, normalize_probability=True):
		sentence_probability_log_sum = 0
		for word in sentence:
			# if word != SENTENCE_START and word != SENTENCE_END:
			word_probability = self.calculate_unigram_probability(word)
			sentence_probability_log_sum += math.log(word_probability, 2)
		return math.pow(2, sentence_probability_log_sum) if normalize_probability else sentence_probability_log_sum
	
	def sorted_vocabulary(self):
		full_vocab = list(self.unigram_frequencies.keys())
		full_vocab.remove(SENTENCE_START)
		full_vocab.remove(SENTENCE_END)
		full_vocab.sort()
		full_vocab.append(UNK)
		full_vocab.append(SENTENCE_START)
		full_vocab.append(SENTENCE_END)
		return full_vocab


def KLdivergence():
	train_stance = pandas.read_csv("fnc-1/split/validataion_stances.csv")
	train_bodies = pandas.read_csv("fnc-1/split/validation_bodies.csv")

	headlines = train_stance["Headline"].values
	bodies = train_bodies["articleBody"].values

	models = UnigramLanguageModel(bodies, smoothing = True)
	KL = 0.0
	for i in range(len(headlines)):
		value1 = models.calculate_sentence_probability(headlines[i], normalize_probability=False)
		value2 = models.calculate_sentence_probability(bodies[i], normalize_probability=False)
	
		if value2 != 0:
			KL = value1 * np.log(value1 / value2)
		else:
			KL = 0
		#result = normalized_mutual_info_score(value1,value2)
		print('headline:')
		print(i)
		print('KL')
		print(KL)



KLdivergence()