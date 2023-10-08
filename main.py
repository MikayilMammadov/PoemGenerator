import pandas as pd
import numpy as np
import tensorflow as tf

# Load the love poems dataset from Excel file
df = pd.read_excel("C:\\Users\\mamma\\OneDrive\\Desktop\\poemFinder.xlsx")
love_poems = df.iloc[:, 0].tolist()
love_poems = [poem for poem in love_poems if isinstance(poem, str)]

# Tokenize the text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(love_poems)
sequences = tokenizer.texts_to_sequences(love_poems)
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences to be of the same length
maxlen = max([len(seq) for seq in sequences])
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='pre')

# Generate input-output pairs
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 50, input_length=seq_length),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=128, epochs=100)

# Generate new love poems
seed_text = "my love for you is like a rose"
for i in range(10):
    # Encode the seed text
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    encoded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length-1, truncating='pre')
    
    # Predict the next word
    yhat = np.argmax(model.predict(encoded), axis=-1)
    
    # Map the integer prediction to a word
    out_word = ''
    for word, index in tokenizer.word_index.items():
        if index == yhat:
            out_word = word
            break
            
    # Append the new word to the seed text
    seed_text += ' ' + out_word
    
    # Output the generated poem
    print(seed_text)