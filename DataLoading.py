import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE
import chess
import chess.pgn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE

data = pd.read_csv('data/games.csv')

grouped_eco_labels = {
    'A00': 'Polish (Sokolsky) opening',
    'A01': 'Nimzovich-Larsen attack',
    'A02-A03': "Bird's opening",
    'A04-A09': 'Reti opening',
    'A10-A39': 'English opening',
    'A40-A41': "Queen's pawn",
    'A42': 'Modern defence, Averbakh system',
    'A43-A44': 'Old Benoni defence',
    'A45-A46': "Queen's pawn game",
    'A47': "Queen's Indian defence",
    'A48-A49': "King's Indian, East Indian defence",
    'A50': "Queen's pawn game",
    'A51-A52': 'Budapest defence',
    'A53-A55': 'Old Indian defence',
    'A56': 'Benoni defence',
    'A57-A59': 'Benko gambit',
    'A60-A79': 'Benoni defence',
    'A80-A99': 'Dutch',
    'B00': "King's pawn opening",
    'B01': 'Scandinavian (centre counter) defence',
    'B02-B05': "Alekhine's defence",
    'B06': 'Robatsch (modern) defence',
    'B07-B09': 'Pirc defence',
    'B10-B19': 'Caro-Kann defence',
    'B20-B99': 'Sicilian defence',
    'C00-C19': 'French defence',
    'C20': "King's pawn game",
    'C21-C22': 'Centre game',
    'C23-C24': "Bishop's opening",
    'C25-C29': 'Vienna game',
    'C30-C39': "King's gambit",
    'C40': "King's knight opening",
    'C41': "Philidor's defence",
    'C42-C43': "Petrov's defence",
    'C44': "King's pawn game",
    'C45': "Scotch game",
    'C46': "Three knights game",
    'C47-C49': "Four knights, Scotch variation",
    'C50': "Italian Game",
    'C51-C52': "Evans gambit",
    'C53-C54': "Giuoco Piano",
    'C55-C59': "Two knights defence",
    'C60-C99': "Ruy Lopez (Spanish opening)",
    'D00': "Queen's pawn game",
    'D01': 'Richter-Veresov attack',
    'D02': "Queen's pawn game",
    'D03': 'Torre attack (Tartakower variation)',
    'D04-D05': "Queen's pawn game",
    'D06': "Queen's Gambit",
    'D07-D09': "Queen's Gambit Declined, Chigorin defence",
    'D10-D15': "Queen's Gambit Declined Slav defence",
    'D16': "Queen's Gambit Declined Slav accepted, Alapin variation",
    'D17-D19': "Queen's Gambit Declined Slav, Czech defence",
    'D20-D29': "Queen's gambit accepted",
    'D30-D42': "Queen's gambit declined",
    'D43-D49': "Queen's Gambit Declined semi-Slav",
    'D50-D69': "Queen's Gambit Declined",
    'D70-D79': 'Neo-Gruenfeld defence',
    'D80-D99': 'Gruenfeld defence',
    'E00': "Queen's pawn game",
    'E01-E09': 'Catalan, closed',
    'E10': "Queen's pawn game",
    'E11': 'Bogo-Indian defence',
    'E12-E19': "Queen's Indian defence",
    'E20-E59': 'Nimzo-Indian defence',
    'E60-E99': "King's Indian defence"
}

def generate_4d_board_states(moves):
    board = chess.Board()
    piece_type_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                      'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    board_states = []
    
    move_list = moves.split()
    move_list = move_list[:28] 

    for move in move_list:
        board.push_san(move)
        board_state = np.zeros((8, 8, 12))
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_type = piece_type_map[str(piece)]
                board_state[rank, file, piece_type] = 1
        board_states.append(board_state)

    return board_states


def map_eco_to_grouped_label(eco_code):
    for key, value in grouped_eco_labels.items():
        if '-' in key:
            start, end = key.split('-')
            if start <= eco_code <= end:
                return value
        elif eco_code == key:
            return value
    return 'Other'

data['grouped_opening'] = data['opening_eco'].apply(map_eco_to_grouped_label)
data = data[data['turns'] >= 28]

grouped_opening_counts = data['grouped_opening'].value_counts()
valid_openings = grouped_opening_counts[grouped_opening_counts >= 10].index
filtered_data = data[data['grouped_opening'].isin(valid_openings)]
data = filtered_data

def balance_dataset_and_split(data, target_column, N_samples, max_count, test_size):
    balanced_data = []
    class_counts = data[target_column].value_counts().to_dict()
    
    for index, row in data.iterrows():
        class_label = row[target_column]
        
        if len(balanced_data) < N_samples:
            balanced_data.append(row)
            class_counts[class_label] -= 1
        else:
            balanced_class_counts = pd.Series([d[target_column] for d in balanced_data]).value_counts()
            max_class = balanced_class_counts.idxmax()

            if balanced_class_counts[max_class] > class_counts[class_label]:
                balanced_data = [d for d in balanced_data if d[target_column] != max_class or balanced_class_counts[max_class] <= class_counts[class_label]]
                balanced_data.append(row)
    
    balanced_df = pd.DataFrame(balanced_data)

    X_train, X_test = train_test_split(balanced_df, test_size=test_size, random_state=42)

    X_train = X_train.drop_duplicates(subset=['moves'])
    X_test = X_test.drop_duplicates(subset=['moves'])

    def upsample_to_balance(data, target_column, max_count):
        groups = data.groupby(target_column)
        upsampled_data = []
        for name, group in groups:
            if len(group) < max_count:
                tiled_group = pd.concat([group] * (max_count // len(group)))
                remainder_group = group.sample(max_count - len(tiled_group), replace=True)
                upsampled_data.append(pd.concat([tiled_group, remainder_group]))
            else:
                upsampled_data.append(group)
        return pd.concat(upsampled_data, ignore_index=True)
    
    X_train = upsample_to_balance(X_train, target_column, max_count)
    
    return X_train, X_test

N_samples = 10000
max_count = 25
test_size = 0.2

X_train, X_test = balance_dataset_and_split(data, 'grouped_opening', N_samples, max_count, test_size)

all_labels = pd.concat([X_train['grouped_opening'], X_test['grouped_opening']])

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

y_train_encoded = label_encoder.transform(X_train['grouped_opening'])
y_test_encoded = label_encoder.transform(X_test['grouped_opening'])

tqdm.pandas()
X_train['board_states_4d'] = X_train['moves'].progress_apply(generate_4d_board_states)
X_train_4d = np.array(X_train['board_states_4d'].to_list())

X_test['board_states_4d'] = X_test['moves'].progress_apply(generate_4d_board_states)
X_test_4d = np.array(X_test['board_states_4d'].to_list())

plt.figure(figsize=(12, 6))
X_train['grouped_opening'].value_counts().plot(kind='bar')
plt.title('Class Distribution in Training Set After Upsampling')
plt.xlabel('Grouped Opening')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.show()

def split_and_save_data(X_train, X_test, y_train, y_test, prefix):
    with open(f'{prefix}_X_train.pkl', 'wb') as f:
        pickle.dump(X_train.astype('float32'), f)
    with open(f'{prefix}_X_test.pkl', 'wb') as f:
        pickle.dump(X_test.astype('float32'), f)
    with open(f'{prefix}_y_train.pkl', 'wb') as f:
        pickle.dump(y_train.astype('float32'), f)
    with open(f'{prefix}_y_test.pkl', 'wb') as f:
        pickle.dump(y_test.astype('float32'), f)

split_and_save_data(X_train_4d, X_test_4d, y_train_encoded, y_test_encoded, prefix='chess')