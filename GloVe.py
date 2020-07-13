from glove import Corpus, Glove

corpus = Corpus() 
sentences = [['나는', '정말', '화난다'], ['너도', '정말', '화나지']]
corpus.fit(sentences, window=5)
# 훈련 데이터로부터 GloVe에서 사용할 동시 등장 행렬 생성

glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
# 학습에 이용할 쓰레드의 개수는 4로 설정, 에포크는 20.

model_result1=glove.most_similar("나는")
print(model_result1)