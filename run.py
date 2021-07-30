from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from graph_construction import graph_construction
from regularizer import regularization
from sklearn.metrics import classification_report

model = SentenceTransformer('bert-base-uncased')
it = 15
mi = 1.0
nb_neighbors = 30
completions = [0.2,0.4,0.6,0.8,1.0]
splits = 5
seed = 0

np.random.seed(seed)

'''
intent:
!git clone https://github.com/karansikka1/documentIntent_emnlp19/ ./dataset/
'''

results = []
for split in range(0, splits):
    for completion in completions:
        df_train = pd.read_json(f'./dataset/splits/train_split_{split}.json')
        df_test = pd.read_json(f'./dataset/splits/val_split_{split}.json')
        
        X_train_im = []
        for index,row in df_train.iterrows():
            features = np.load('./resnet18_feat/'+row['id']+'.npy')
            X_train_im.append(features)
        X_train_im = np.array(X_train_im)

        X_test_im = []
        for index,row in df_test.iterrows():
            features = np.load('./resnet18_feat/'+row['id']+'.npy')
            X_test_im.append(features)
        X_test_im = np.array(X_test_im)

        y_train = np.array(df_train['intent'])
        y_test = np.array(df_test['intent'])

        X_train_txt = model.encode(df_train['caption'])
        X_test_txt = model.encode(df_test['caption'])

        incomplete_indices = np.random.randint(0, len(y_test), int(len(y_test) * (1.0-completion)))
        incomplete_indices += len(y_train)

        X_txt = np.concatenate([X_train_txt, X_test_txt], axis=0)
        X_img = np.concatenate([X_train_im, X_test_im], axis=0)
        S = graph_construction(X_txt, X_img, incomplete_indices, k=nb_neighbors)
        X_prop = regularization(S, nb_neighbors, X_txt, incomplete_indices, iterations=it, mi=mi)

        X_train = np.concatenate([X_prop[:len(y_train)], X_train_im], axis=1)
        X_test = np.concatenate([X_prop[len(y_train):], X_test_im], axis=1)

        clf = SVC(gamma='auto', probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)

        print(classification_report(y_test, y_pred))