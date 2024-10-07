import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Input # type: ignore
from keras.metrics import F1Score # type: ignore
from IPython.display import display, HTML

def evaluate_model(model, model_type, X_train, y_train, X_test, y_test, fit_params={}):
    if model_type == 'sklearn':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == 'keras':
        display(HTML("<h2>Training:</h2>"))
        history = model.fit(X_train, y_train, **fit_params)
        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # Plot loss and f1 history
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(history.history['loss'], label='Training Loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Weighted F1 Score', color='tab:orange')
        if 'f1_score' in history.history:
            f1_scores = history.history['f1_score']
            ax2.plot(f1_scores, label='Training Weighted F1 Score', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, 1)

        fig.tight_layout()
        plt.title('Loss and Weighted F1 Score History')
        fig.legend(loc='upper left')
        plt.show()

        display(HTML("<h2>Testing:</h2>"))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")


    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()

    return

# CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=input_shape))
    
    # Convolutional layers applied to each time step
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    
    # Flatten the time step dimension
    model.add(Flatten())  # Flattening both the time steps and features
    
    # Dense layer to match number of classes
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', F1Score(average='weighted', threshold=None, name="f1_score", dtype=None)])
    
    return model

# LSTM Model
def create_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # LSTM layers
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', F1Score(average='weighted', threshold=None, name="f1_score", dtype=None)])
    return model

# Combined CNN-LSTM Model
def create_combined_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # CNN layers
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    
    # Flatten CNN output
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layers
    model.add(LSTM(100))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', F1Score(average='weighted', threshold=None, name="f1_score", dtype=None)])
    
    return model

# Combined CNN-LSTM Model with Attention
from Attention import CustomAttention
from keras.layers import Reshape, SpatialDropout1D, Bidirectional, Dropout # type: ignore
from keras.models import Model # type: ignore

def create_combined_cnn_lstm_with_attention(input_shape, num_classes, maxlen):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # CNN layers
    model.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    # Flatten CNN output
    model.add(TimeDistributed(Flatten()))
    model.add(Reshape((maxlen, -1)))

    # LSTM with Bidirectional wrapper
    model.add(Bidirectional(LSTM(50, return_sequences=True)))

    # Attention Layer
    model.add(CustomAttention(step_dim=maxlen))

    # Dense layers
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Create Model
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', F1Score(average='weighted', threshold=None, name="f1_score", dtype=None)])
    
    return model