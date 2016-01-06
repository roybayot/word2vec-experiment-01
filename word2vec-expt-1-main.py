import gensim

model_file = "new.glove.twitter.27B.25d.txt"
model_file = "test.txt"
model=gensim.models.Word2Vec.load_word2vec_format(model_file,binary=False) #GloVe Model

print model.most_similar(positive=['australia'], topn=10)
print model.similarity('woman', 'man')
