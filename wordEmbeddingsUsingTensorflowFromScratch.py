import numpy as np
import tensorflow as tf

corpus_raw = 'He is the king . The king is royal . She is the royal  queen '

# ---------------------------------------------------------------------------------------------------
# data preprocessing in required form
# ---------------------------------------------------------------------------------------------------

corpus_raw = corpus_raw.lower()
words = []
for word in corpus_raw.split():
	if word != '.': # because we donr want to treat '.' as a word
		words.append(word)

words = set(words)  # removes duplicates
word2int = {}
int2word = {}
vocab_size = len(words)  # gives the total number of unique words

for i,word in enumerate(words):
	word2int[word] = i
	int2word[i] = word
print(word2int)  # every word has a unique int associated with it
# -> {'the': 1, 'king': 2, 'she': 0, 'is': 3, 'royal': 4, 'he': 5, 'queen': 6}

raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())
# print(sentences)
# -> [['he', 'is', 'the', 'king'], ['the', 'king', 'is', 'royal'], ['she', 'is', 'the', 'royal', 'queen']]


# ---------------------------------------------------------------------------------------------------
# creating the skip gram model i.e getting the adjacent word pairs in the given window size
# ---------------------------------------------------------------------------------------------------

data = []
WINDOW_SIZE = 2

for sentence in sentences:
	for word_index,word in enumerate(sentence):
		for nb_word in sentence[max(word_index-WINDOW_SIZE+1,0):min(word_index+WINDOW_SIZE,len(sentence))+1]:
			if nb_word!=word:
				data.append([word,nb_word])

# print(data)
# gives the word pairs
# -> [['he', 'is'], ['he', 'the'], ['is', 'he'], ['is', 'the'], ['is', 'king'], ['the', 'is'], ['the', 'king'], ['king', 'the'], ['the', 'king'], ['the', 'is'], ['king', 'the'], ['king', 'is'], ['king', 'royal'], ['is', 'king'], ['is', 'royal'], ['royal', 'is'], ['she', 'is'], ['she', 'the'], ['is', 'she'], ['is', 'the'], ['is', 'royal'], ['the', 'is'], ['the', 'royal'], ['the', 'queen'], ['royal', 'the'], ['royal', 'queen'], ['queen', 'royal']]


def to_one_hot(data_point_index,vocab_size):  # gives the one hot vector of required format
	temp = np.zeros(vocab_size)
	temp[data_point_index]=1
	return temp

x_train = [] # input word
y_train = [] # output word

#create one big 2 d matrix apending all one hot vectors
for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# print(x_train.shape, y_train.shape)
# -> (27, 7) (27, 7)   # meaning 27 training points, where each point has 7 dimensions



# ---------------------------------------------------------------------------------------------------
# Creating the tensorflow model  # making placeholders for x_train and y_train
# ---------------------------------------------------------------------------------------------------

# variables and placeholders which are used in the computation graph of tensorflow
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5 # you can choose your own number - this is the final dimension of the word vectors that we want  --- this is similar to pca,it takes the significant 5 dimensions after minimizing the loss
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)
# print(hidden_representation)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
print(prediction)


# ---------------------------------------------------------------------------------------------------
# Training the model starts here
# ---------------------------------------------------------------------------------------------------
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 1000 
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))


vectors = sess.run(W1 + b1)
#final computed vectors
print(vectors)
# print("the vector of queen is " , vectors[ word2int['queen'] ])



# ---------------------------------------------------------------------------------------------------
# getting the closest word for a given word - comparing the distance between them
# ---------------------------------------------------------------------------------------------------
def euclidean_dist(a,b):
	return np.linalg.norm(a-b)

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print(int2word[find_closest(word2int['king'], vectors)])
print(int2word[find_closest(word2int['queen'], vectors)])
print(int2word[find_closest(word2int['royal'], vectors)])


# ---------------------------------------------------------------------------------------------------
# converting them into 2 dimensions and plotting them using tSNE
# ---------------------------------------------------------------------------------------------------
from sklearn import preprocessing
import matplotlib.pyplot as plt

vectors = preprocessing.normalize(vectors, norm='l2')
fig, ax = plt.subplots()
print(words)
for word in words:
    print(word, vectors[word2int[word]][1])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))
plt.show()