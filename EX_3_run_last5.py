
# install packages
# !pip install jiwer
# !pip install pandas
# !pip install matplotlib
# !pip install sentence-



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

import sys
import time
from tqdm import tqdm
import pickle
import random

import torch
# device = torch.device("cpu")
# print(device)

from sentence_transformers import SentenceTransformer, util

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# model_name = 'sentence-transformers/all-distilroberta-v1'
minilm = SentenceTransformer(model_name)
# minilm = SentenceTransformer(model_name, device=device)

seed = 2
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ## Data Prep
# ### Load Data

data_path = "/home/slimlab/Downloads/LJSpeech-1.1"
wavs_path = data_path + "/16khz/wavs/"
metadata_path = data_path + "/metadata.csv"

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
metadata_df.head(3)


# ### Split Data


split = int(len(metadata_df) * 0.90)
df_train = metadata_df[:split]
df_val = metadata_df[split:]

print(f"Size of the training set: {len(df_train)}")
print(f"Size of the training set: {len(df_val)}")


# ### Character ↔️ Number Function

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")

num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)

print(tf.config.list_physical_devices('GPU'))


# ### Data encoding function

from tensorflow.python.ops.gen_spectral_ops import fft
# int. scalar tensor. window length in samples
frame_length = 256
# int. scalar tensor. number of samples to step
frame_step = 160
# int scalar tensor. size of FFT
fft_length = 384

def encode_single_sample(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wavs_path + wav_file + ".wav")

    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float, this step might not be necessary
    audio = tf.cast(audio, tf.float32)

    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, 
        frame_length=frame_length, 
        frame_step=frame_step, 
        fft_length=fft_length
    )

    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    label = tf.strings.lower(label)

    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")

    # 9. Map the characters in label to numbers
    label = char_to_num(label)

    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label



batch_size = 16

# define training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
)

train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# ## Functions

# In[11]:


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# In[12]:


# embeds and calculates cosine similarity
# returns matrix of all cosine similaritys
# between s and s1
@torch.no_grad()
def get_cos_sim(s, s1, model=minilm):
    embedding_s = model.encode(s, convert_to_tensor=True)
    embedding_s1 = model.encode(s1, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_s, embedding_s1)


# get the diagonals of the cosine matrix
@torch.no_grad()
def get_respective_cos_sim(s, s1, model=minilm):
    cos_sim = get_cos_sim(s, s1, model)
    return cos_sim.diagonal()

# smoothing relu so that there will never be
# log of negative number or 0
@torch.no_grad()
def relu(x):
    return torch.max(torch.tensor(0.0000001),x)

# custom cosine loss
# the negative log of the cosine similarity
@torch.no_grad()
def get_cos_loss(s, s1, model=minilm):
    cos_sim = get_respective_cos_sim(s, s1, model)
    # remove negatives and look at neg loglikilood
    cos_sim = relu(cos_sim).reshape((len(s),1))
    return -np.log(cos_sim.cpu()).numpy()

@torch.no_grad()
def get_cos_distance(s, s1, model=minilm):
    cos_sim = get_respective_cos_sim(s, s1, model)
    # remove negatives and look at neg loglikilood
    cos_sim = relu(cos_sim).reshape((len(s),1))
    return 1 - cos_sim.cpu().numpy()

# decodes set of label
def get_labels(y):
    gt = []
    for label in y:
        label = (
            tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        )
        gt.append(label)
    return gt


# In[13]:


# Normal CTCLoss function
@torch.no_grad()
def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# In[14]:

@torch.no_grad()
def CTC_Cosine_Loss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    y_true_labs = get_labels(y_true)
    # print(y_true_labs)
    y_pred_labs = decode_batch_predictions(y_pred)
    # print(y_pred_labs)
    cos_loss = get_cos_loss(y_true_labs, y_pred_labs)
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    # print(cos_loss)
    return loss + cos_loss

@torch.no_grad()
def CTC001_Cosine_Loss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    y_true_labs = get_labels(y_true)
    # print(y_true_labs)
    y_pred_labs = decode_batch_predictions(y_pred)
    # print(y_pred_labs)
    cos_loss = get_cos_loss(y_true_labs, y_pred_labs)
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    # print(cos_loss)
    return (0.001 *loss) + cos_loss

@torch.no_grad()
def CTC_Cosine2_Loss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    y_true_labs = get_labels(y_true)
    # print(y_true_labs)
    y_pred_labs = decode_batch_predictions(y_pred)
    # print(y_pred_labs)
    cos_loss = get_cos_loss(y_true_labs, y_pred_labs)
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    # print(cos_loss)
    return loss + tf.math.square(cos_loss)




@torch.no_grad()
def CTC_by_Cosine_Dist(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    y_true_labs = get_labels(y_true)
    # print(y_true_labs)
    y_pred_labs = decode_batch_predictions(y_pred)
    # print(y_pred_labs)
    cos_distance = get_cos_distance(y_true_labs, y_pred_labs)
    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    # print(cos_loss)
    return loss * cos_distance

# ## Model

# In[15]:


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128, loss=CTCLoss):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax", name="output")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.legacy.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=loss, run_eagerly=True)
#     model.compile(optimizer=opt, loss=loss)
    
    return model





# In[16]:
rnnu = 512

if sys.argv[1] == 'base':

    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnnu,
        loss=CTCLoss,
    )

elif sys.argv[1] == 'cos':
    
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnnu,
        loss=CTC_Cosine_Loss,
    )

elif sys.argv[1] == 'ctc001':
    
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnnu,
        loss=CTC001_Cosine_Loss,
    )

elif sys.argv[1] == 'cossquared':
    
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnnu,
        loss=CTC_Cosine2_Loss,
    )

elif sys.argv[1] == 'ctcbycos':
    
    model = build_model(
        input_dim=fft_length // 2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnnu,
        loss=CTC_by_Cosine_Dist,
    )
    

else:
    print('put either base or cos as arguments')
    exit(1)

model.summary()


# In[34]:

model.load_weights('model_base_weights_42')


def train(model, epochs, name='model'):
    hist = {
        'losses': [],
        
        'train_wer': [],
        # 'train_cos_loss': [],
        'train_cos_dis': [],
        
        'val_wer': [],        
        # 'val_cos_loss': [],
        'val_cos_dis': []
    }
    
    for epoch in tqdm(range(43,50)):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        if epoch > 44:
            for step, (X,y) in enumerate(tqdm(train_dataset)):
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = model(X)
                    loss = CTCLoss(y, y_pred)

                    hist['losses'].extend(loss.numpy().reshape(1, len(loss)).tolist()[0])

                # Calculate gradients with respect to every trainable variable
                grad = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grad, model.trainable_weights))

                if step % 200 == 0:
                    print('avg. loss at step: ', step, ' = ', loss.numpy().mean())
        else:
            for step, (X,y) in enumerate(tqdm(train_dataset)):
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = model(X)
                    loss = model.loss(y, y_pred)

                    hist['losses'].extend(loss.numpy().reshape(1, len(loss)).tolist()[0])

                # Calculate gradients with respect to every trainable variable
                grad = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grad, model.trainable_weights))

                if step % 200 == 0:
                    print('avg. loss at step: ', step, ' = ', loss.numpy().mean())
        
        
        train_wer, train_cos_dis = get_performance(model, train_dataset)
        val_wer, val_cos_dis = get_performance(model, validation_dataset)
        
        hist['train_wer'].append(train_wer)
        # hist['train_cos_loss'].append(train_cos_loss)
        hist['train_cos_dis'].append(train_cos_dis)
        
        hist['val_wer'].append(val_wer)
        # hist['val_cos_loss'].append(val_cos_loss)
        hist['val_cos_dis'].append(val_cos_dis)
        
        if epoch % 3 == 0:
            model.save_weights(name + '_weights_' + str(epoch))
        print('Time: ', time.time() - start_time)
    return hist



# In[35]:

@torch.no_grad()
def get_performance(model, data):
    with torch.no_grad():
        print('getting performance')
        predictions = []
        targets = []
        cos_losses = []
        cos_distances = []

        # for step, (X,y) in enumerate(tqdm(data)):
        for batch in tqdm(data):
            X, y = batch

            # make prediction
            batch_predictions = model(X, training = False)

            # convert prediction and target into text
            batch_predictions = decode_batch_predictions(batch_predictions)
            temp_target = get_labels(y)

            # add text to lists
            predictions.extend(batch_predictions)
            targets.extend(temp_target)
        
            # get cosine loss
            # cos_losses.extend(get_cos_loss(temp_target, batch_predictions).reshape(1, len(temp_target)).tolist()[0])
            # torch.cuda.empty_cache()
            # get average cosine distance (1 - cos(x,y))
            cos_distances.extend((1 - get_respective_cos_sim(temp_target, batch_predictions)).tolist())        
            torch.cuda.empty_cache()

        # get metrics
        wer_score = wer(targets, predictions)
        # avg_cos_loss = np.mean(cos_losses)
        avg_cos_distance = np.mean(cos_distances)

        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        # print(f"Avg. Cos Loss: {avg_cos_loss:.4f}")
        print(f"Avg. Cos Distance: {avg_cos_distance:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print(f"Word Error Rate For Example: {wer(targets[i], predictions[i]):.4f}")
            print(f"Cos Dis: {1 - get_cos_sim(targets[i], predictions[i]).item():.4f}")
            print("-" * 100)
    return wer_score, avg_cos_distance

# In[136]:


history_data = train(model, epochs=50, name='model_'+ sys.argv[1])


# In[ ]:


with open('history_' + str(rnnu) + '_'+ sys.argv[1] +'_seed2.pkl', 'wb') as handle:
    pickle.dump(history_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


