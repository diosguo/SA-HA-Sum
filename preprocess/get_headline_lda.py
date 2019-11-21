from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from tqdm import tqdm


n_topics = 32

tfidf_vectorizer = TfidfVectorizer(min_df=2)

lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=10)

path_to_headlines = '/home/xuyang/data/headlines'

file_list = os.listdir(path_to_headlines)
datas = []
for filename in tqdm(file_list):
    with open(os.path.join(path_to_headlines, filename), 'r', encoding='windows-1253') as f:
        try:
            datas.append(f.readline())
        except UnicodeDecodeError as u:
            print(filename)
            raise u

# fit tfidf
tfidf_mat = tfidf_vectorizer.fit_transform(datas)

# fit lda_model
lda_mat = lda_model.fit_transform(tfidf_mat)

pickle.dump(tfidf_mat, open('../data/tfidf_model.pkl','wb'))
pickle.dump(lda_model, open('../data/lda_model.pkl','wb'))

for idx, filename in enumerate(file_list):
    with open(os.path.join('/home/xuyang/data/headline_lda', filename),'wb') as f:
        pickle.dump(lda_mat[idx].tolist(), f)
    


