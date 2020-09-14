import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from tensorflow.keras.layers import *
from tensorflow.python.ops import init_ops
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="0"

with open('./output/train_input.pkl', 'rb') as f:
    train_input = pkl.load( f)
with open('./output/train_output.pkl', 'rb') as f:
    train_output=pkl.load( f)
with open('./output/val_input.pkl', 'rb') as f:
    val_input=pkl.load(f)
with open('./output/val_output.pkl', 'rb') as f:
    val_output=pkl.load(f)
with open('./output/test_input.pkl', 'rb') as f:
    test_input=pkl.load(f)
with open('./output/test_output.pkl', 'rb') as f:
    test_output=pkl.load(f)
with open('./output/inp_w2i.pkl', 'rb') as f:
    inp_word2idx=pkl.load(f)
with open('./output/inp_i2w.pkl', 'rb') as f:
    inp_idx2word=pkl.load(f)
with open('./output/oup_w2i.pkl', 'rb') as f:
    oup_word2idx=pkl.load(f)
with open('./output/oup_i2w.pkl', 'rb') as f:
    oup_idx2word=pkl.load(f)
with open('./output/inp_wm.pkl', 'rb') as f:
    inp_matrix = pkl.load(f)
with open('./output/oup_wm.pkl', 'rb') as f:
    oup_matrix=pkl.load(f)

input_vocab_size = len(inp_idx2word)
target_vocab_size = len(oup_word2idx)

num_of_head = 4
hidden_dim = 256
num_layers = 3
dropout_ratio = 0.1
BATCH_SIZE = 256
Epochs = 5
embedding_dim = 256

print('input word_vector matrix shape : ', inp_matrix.shape)
print('output word_vector matrix shape : ', oup_matrix.shape)
train_input = train_input[:1500000]
train_output = train_output[:1500000]
val_input = val_input[:500000]
val_output = val_output[:500000]
test_input = test_input[:500000]
test_output = test_output[:500000]
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dropout_ratio):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi_lstm = Bidirectional(LSTM(hidden_dim//2 , return_sequences = True, return_state=True, dropout = dropout_ratio), merge_mode = 'concat')
        self.dropout = Dropout(rate=dropout_ratio)
        
    def call(self, x, hidden_state):
        bi_lstm_output, state_h_f, state_c_f, state_h_b, state_c_b = self.bi_lstm(x, initial_state = hidden_state)
#         state_h_f : forward LSTM hidden state
#         state_c_f : forward LSTM cell state
#         state_h_b : backward LSTM hidden state
#         state_c_b : backward LSTM cell state
        hidden_state = Concatenate()([state_h_f,state_h_b])
        cell_state = Concatenate()([state_c_f, state_c_b])
        
        dropout_output = self.dropout(bi_lstm_output)
        
        add_output = x + dropout_output
        return add_output, hidden_state, cell_state

class Encoder(tf.keras.models.Model):
    def __init__(self, input_vocab_size, num_layers, embedding_dimension, hidden_dimension, dropout_ratio, embedding_matrix = None):
        super(Encoder, self).__init__()
        if embedding_dimension // hidden_dimension != 2:
            assert 'Embedding dimension have to twice hidden size.'
        self.num_layers = num_layers
        self.hidden_dim = hidden_dimension
        self.output_dim = self.hidden_dim
        # if all(embedding_matrix):
        self.embedding = Embedding(input_vocab_size, embedding_dimension, weights = [embedding_matrix], trainable = False)
        # else:
            # self.embedding = Embedding(input_vocab_size, embedding_dimension)
        self.dropout = Dropout(dropout_ratio, name = 'encoder dropout')
        self.encoder_layers = [EncoderLayer(self.hidden_dim, dropout_ratio) for i in range(num_layers)]
        self.projeciton_layer = Dense(self.output_dim, activation = 'linear')
        
    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        hidden_state_list = []
        enc_output = x
        for i in range(num_layers):
            enc_output, hidden_state, cell_state = self.encoder_layers[i](enc_output, hidden)
            hidden_state_list.append([hidden_state, cell_state])
        # calcualate encoder layer
        
        enc_output = self.projeciton_layer(enc_output)
        # linear activation layer(projection layer) 
        return enc_output, hidden_state_list
    
    def make_initial_state(self, batch_size):
        zero_state = tf.zeros(shape = (batch_size, self.hidden_dim//2))
        return [zero_state, zero_state, zero_state, zero_state]


class MultiHead_Addictive(tf.keras.layers.Layer):
    def __init__(self, head_num, dimension, use_scale = True):
        super(MultiHead_Addictive, self).__init__()
        self.head_num = head_num
        self.use_scale = use_scale
        if self.use_scale:
            self.scale = self.add_weight(shape = [dimension//head_num],
                                        initializer= init_ops.glorot_uniform_initializer(),
                                        trainable=True,
                                        dtype = self.dtype)
        else:
            self.scale = None
        # define scale value
        
    def split_head(self, query, key):
        batch_size = query.shape[0]
        # define batch_size
        
        query_length = query.shape[1]
        key_length = key.shape[1]
        # define data length
    
        query = tf.reshape(query, shape = (batch_size, query_length, self.head_num, -1))
        key = tf.reshape(key, shape =  (batch_size, key_length, self.head_num, -1))
        
        query = tf.transpose(query, perm=[0,2,1,3])
        key = tf.transpose(key, perm = [0,2,1,3])
        # split head
        return query, key
    
    def AdditiveAttention(self, query, key):
        # print('query key shape',query.shape, key.shape)
        # print(self.scale.shape)
        # print(tf.reduce_sum(tf.tanh(query + key), axis = -1).shape)
        if self.use_scale:
            attention_value = tf.reduce_sum(self.scale * tf.tanh(query + key), axis = -1)
        else:
            attention_value = tf.reduce_sum(tf.tanh(query + key), axis = -1)
            
        attention_value = tf.expand_dims(attention_value, axis = -1)
        context_value = attention_value * query
        return context_value
    
    def call(self, query, key):
        query, key = self.split_head(query, key)
        multihead_output = []
        for i in range(self.head_num):
            multihead_output.append(self.AdditiveAttention(query[::,i], key[::, i]))
        multihead_output = Concatenate()(multihead_output)
        multihead_output = tf.expand_dims(tf.reduce_sum(multihead_output, axis=1), axis = 1)
        
        return multihead_output

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, dropout_ratio):
        super(DecoderLayer, self).__init__()
        self.concat = Concatenate()
        self.lstm = LSTM(hidden_dim, return_sequences=True, return_state=True,dropout = dropout_ratio)
        self.dropout = Dropout(dropout_ratio)
        
    def call(self, x, context_vector, state):
        concat_output = self.concat([x, context_vector])
        lstm_output, h_state, c_state = self.lstm(concat_output, initial_state = state)
#         print(lstm_output.shape)
        lstm_output = self.dropout(lstm_output)
        add_output = lstm_output + x
        return add_output, h_state, c_state

class Decoder(tf.keras.models.Model):
    def __init__(self, target_vocab_size, embedding_dimension, head_num, num_layers, hidden_dim, dropout_value, embedding_matrix = None):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        # if all(embedding_matrix):
        self.embedding = Embedding(target_vocab_size, embedding_dimension, weights = [embedding_matrix], trainable = False)
        # else:
            # self.embedding = Embedding(target_vocab_size, embedding_dimension)
        self.lstm = LSTM(hidden_dim, return_sequences = True, return_state = True, dropout = dropout_ratio)
        self.dropout = Dropout(dropout_ratio)
        self.multihead_attention = MultiHead_Addictive(head_num, hidden_dim)
        self.decoder_layers = [DecoderLayer(hidden_dim, dropout_value) for _ in range(num_layers)]
        self.output_dense = Dense(target_vocab_size)
    
    def call(self, x, encoder_output, lstm_state, layer_state):
        x = self.embedding(x)
        dropout_x = self.dropout(x)
        
        context_vector =  self.multihead_attention(encoder_output, tf.expand_dims(lstm_state[0], axis = -2))
#         get context vector
        lstm_input = Concatenate()([dropout_x, context_vector])
#         lstm_input : concatenate( embedding_value , context_vector)
        lstm_output, lstm_hidden_state, lstm_cell_state = self.lstm(lstm_input, initial_state = lstm_state)
#         initialstate : add output(encoder lstm output)
        new_lstm_state = [lstm_hidden_state, lstm_cell_state]
#         new initial state : lstm_hidden_state, lstm_cell_state

        new_layer_state = []
        f_lstm_input = lstm_output
        for i in range(self.num_layers):
            f_lstm_input, hidden, cell = self.decoder_layers[i](f_lstm_input, context_vector, layer_state[i])
            new_layer_state.append([hidden, cell])
        output = self.output_dense(f_lstm_input)
        # print(output.shape)
        return output, new_lstm_state, new_layer_state



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(hidden_dim)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name = 'train_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

def train_step(inp, tar, encoder_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, hidden_state = encoder(inp, encoder_hidden)
        dec_lstm_state = [(hidden_state[0][0] + hidden_state[1][0] + hidden_state[2][0])/3,
                          (hidden_state[0][1] + hidden_state[1][1] + hidden_state[2][1])/3]
        
        # print(dec_lstm_state[0].shape, dec_lstm_state[1].shape)
        target_token = tf.expand_dims(tar[:, 0], 1)
        for t in range(1, tar.shape[1]):
            predictions, dec_lstm, dec_layer_lstm = decoder(target_token,
                                                            enc_output,
                                                            dec_lstm_state,
                                                            hidden_state)
            loss += loss_function(tar[:, t], predictions)    
            target_token = tf.expand_dims(tar[:, t] , 1)
        batch_loss = (loss/int(tar.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss        

def val_step(inp, tar, encoder_hidden):
    loss = 0
    
    enc_output, hidden_state = encoder(inp, encoder_hidden)
    dec_lstm_state = [(hidden_state[0][0] + hidden_state[1][0] + hidden_state[2][0])/3,
                      (hidden_state[0][1] + hidden_state[1][1] + hidden_state[2][1])/3]
    
    # print(dec_lstm_state[0].shape, dec_lstm_state[1].shape)
    target_token = tf.expand_dims(tar[:, 0], 1)
    for t in range(1, tar.shape[1]):
        # print(target_token.shape)
        predictions, dec_lstm, dec_layer_lstm = decoder(target_token,
                                                        enc_output,
                                                        dec_lstm_state,
                                                        hidden_state)
        loss += loss_function(tar[:, t], predictions)   

        target_token = tf.argmax(predictions, -1)
    batch_loss = (loss/int(tar.shape[1]))
    return batch_loss
encoder = Encoder(len(inp_idx2word), num_layers, hidden_dim  , hidden_dim, dropout_ratio, inp_matrix)
decoder = Decoder(target_vocab_size = target_vocab_size, 
                  embedding_dimension = embedding_dim, 
                  head_num = num_of_head, 
                  num_layers = 3, 
                  hidden_dim = hidden_dim, 
                  dropout_value = 0.1,
                  embedding_matrix = oup_matrix)

print('target_vocab_size = ',target_vocab_size, 
                  'embedding_dimension =', embedding_dim, 
                  'head_num = ',num_of_head, 
                  'num_layers =', 3, 
                  'hidden_dim =', hidden_dim, 
                  'dropout_value =', 0.1,
                  'embedding_matrix =', oup_matrix.shape)
checkpoint_dir = './weight'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

batch_loss = train_step(train_input[:BATCH_SIZE], 
                        train_output[:BATCH_SIZE], 
                        encoder.make_initial_state(BATCH_SIZE))

steps_per_epoch = len(train_input)//BATCH_SIZE
train_epoch_loss = []
val_epoch_loss = []
val_steps_per_epoch = len(val_input)//BATCH_SIZE
print('iter per epoch : {}'.format(steps_per_epoch))
#### read_weight ####

initial_state = encoder.make_initial_state(256)
encoder_output,hidden_state= encoder(train_input[:256], initial_state)
encoder.load_weights('./final_encoder.h5')
example_target_input = train_output[:256, :1]
encoder_output = tf.ones(shape =  (256,32,256))
lstm_state = [tf.zeros((256, 256)), tf.zeros((256, 256))]
layer_state = [lstm_state, lstm_state, lstm_state]
output, new_lstm_state, new_layer_state = decoder(example_target_input, 
                                                  encoder_output, 
                                                  lstm_state, 
                                                  layer_state)
decoder.load_weights('./final_decoder.h5')

for e in range(Epochs):
    print('{} epoch training start'.format(e+1))
    encoder_hidden = encoder.make_initial_state(BATCH_SIZE)
    total_loss = 0

    for batch in tqdm(range(steps_per_epoch)):
        batch_input = train_input[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        batch_output = train_output[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]

        batch_loss = train_step(batch_input,  batch_output, encoder_hidden)

        total_loss += batch_loss.numpy()

        if (batch+1) % 1000 == 0:
            print('batch : {} Loss : {:.4f}'.format(batch+1, total_loss/batch))

    val_loss = 0
    for batch in tqdm(range(len(val_input)//BATCH_SIZE)):
        batch_input = val_input[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        batch_output = val_input[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
        batch_loss = val_step(batch_input,  batch_output, encoder_hidden)

        val_loss += batch_loss.numpy()

    total_loss = float(total_loss)
    val_loss = float(val_loss)
    train_epoch_loss.append(total_loss/steps_per_epoch)
    print('Epoch : {} Loss : {:.4f}'.format(e+1, total_loss/steps_per_epoch))
    print('Epoch : {} validation Loss : {:.4f}'.format(e+1, val_loss/val_steps_per_epoch))
    if len(val_epoch_loss) == 0:
        pass
    elif min(val_epoch_loss)>val_loss/val_steps_per_epoch:
        print('{} epoch validation loss : {:.4f} \n {} epoch validation loss : {:.4f}'.format(e, val_epoch_loss[e-1], e+1, val_loss/val_steps_per_epoch))
        print('save success')
        encoder.save_weights('encoder.h5')
        decoder.save_weights('decoder.h5')
    val_epoch_loss.append(val_loss/val_steps_per_epoch)

encoder.save_weights('final_encoder.h5')
decoder.save_weights('final_decoder.h5')
def test_step(inp, tar, encoder_hidden):
    enc_output, hidden_state = encoder(inp, encoder_hidden)
    dec_lstm_state = [(hidden_state[0][0] + hidden_state[1][0] + hidden_state[2][0])/3,
                      (hidden_state[0][1] + hidden_state[1][1] + hidden_state[2][1])/3]
    output = []
    target_token = tf.expand_dims(tar[:, 0], 1)
    loss = 0
    for t in range(1, tar.shape[1]):
        predictions, dec_lstm, dec_layer_lstm = decoder(target_token,
                                                        enc_output,
                                                        dec_lstm_state,
                                                        hidden_state)
        loss += loss_function(tar[:, t], predictions)
        predictions = tf.argmax(predictions, -1)
        output.append(predictions)
        target_token = predictions
    batch_loss = (loss/int(tar.shape[1]))
    output = tf.concat(output, axis = -1)
    return output, batch_loss.numpy()

f = open(output_text.txt, 'w')
for batch in range(len(test_input)//BATCH_SIZE):
    batch_input = test_input[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
    batch_output = test_input[batch * BATCH_SIZE : (batch + 1) * BATCH_SIZE]
    output, batch_loss = test_step(batch_input,  batch_output, encoder_hidden)
    for word in output[0]:
        print(oup_idx2word[word.numpy()], end = ' ')
    for sent in output:
        for word in sent:
            f.write(oup_idx2word[word.numpy()] + ' ')
        f.write('\n')
f.close()
