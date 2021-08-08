from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from graph_construction import graph_construction
from regularizer import regularization
from sklearn.metrics import classification_report

it = 15
mi = 1.0
nb_neighbors = 30
completions = [1.0,0.8,0.6,0.4,0.2]
splits = 5
seed = 0

np.random.seed(seed)

'''
intent:
!git clone https://github.com/karansikka1/documentIntent_emnlp19/ ./dataset/
!wget https://www.dropbox.com/s/pp1nkipzklrgqwl/paper-intent.zip
!tar -xvf ./dataset/resnet18_feat.tar
!unzip paper-intent.zip -d ./features
'''

labels = {"provoke":0, "inform":1, 
        "advocate":2, "entertain":3, 
        "expose":4, "express":5, 
        "promote":6}

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

        y_train = np.load(f'./features/train_target_{split}.npy')
        y_train = np.argmax(y_train, axis=1)
        y_test = np.load(f'./features/test_target_{split}.npy')
        y_test = np.argmax(y_test, axis=1)

        incomplete_indices = np.random.randint(0, len(y_test), int(len(y_test) * (1.0-completion)))
        incomplete_indices += len(y_train)

        X_train_txt = np.load(f'./features/train_fts_{split}.npy')
        X_test_txt = np.load(f'./features/test_fts_{split}.npy')
        
        X_txt = np.concatenate([X_train_txt, X_test_txt], axis=0)

        for inc in incomplete_indices:
            X_txt[inc] *= 0.0

        X_img = np.concatenate([X_train_im, X_test_im], axis=0)
        S = graph_construction(X_txt, X_img, incomplete_indices, k=nb_neighbors)
        X_prop = regularization(S, nb_neighbors, X_txt, incomplete_indices, iterations=it, mi=mi)

        X_train = np.concatenate([X_prop[:len(y_train)], X_train_im], axis=1)
        X_test = np.concatenate([X_prop[len(y_train):], X_test_im], axis=1)

        clf = SVC(gamma='auto', probability=True)
        clf.fit(X_train, y_train)
        y_probs = clf.predict_proba(X_test)
        y_pred = np.argmax(y_probs, axis=1)

        result = {
            'split':split,
            'completion': completion,
            'acc': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_probs, multi_class='ovr')
        }
        print(result)
        results.append(result)

df_results = pd.DataFrame(results)
df_results.to_csv('results_prop.csv')
        