from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os


n_topics = 32

tfidf_vectorizer = TfidfVectorizer(min_df=2)

lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10)

file_list = os.path.join('../data/cnn_headline/')
datas = []
for filename in file_list:
    with open(os.path.join('../data/cnn_headline/', filename), 'r') as f:
        datas.append(f.readline())

tfidf_mat = tfidf_vectorizer.fit_transform(datas)
lda_mat = lda_model.fit_transform(tfidf_mat)

pickle.dump(tfidf_mat, open('../data/tfidf_model.pkl','wb'))
pickle.dump(lda_model, open('../data/lda_model.pkl','wb'))

for idx, filename in enumerate(file_list):
    with open(os.path.join('../data/cnn_head_lda/', filename),'wb') as f:
        pickle.dump(lda_mat[idx].tolist(), f)
    


