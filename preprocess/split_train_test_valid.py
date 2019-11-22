import hashlib
import os
import shutil
from tqdm import tqdm
def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode('utf-8'))
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]

cate = ['cnn']
data_type = ['test','training','validation']

# for c in cate:
#     for d in data_type:
#         path = '_'.join([c,'wayback',d,'urls'])+'.txt'
#         file_list = read_text_file(os.path.join('url_lists',path))
#         hashs = get_url_hashes(file_list)

#         print(hashs[0]+'stry')
#         print(path)
        
file_list = read_text_file(os.path.join('url_lists','cnn_wayback_test_urls.txt'))
hashs = get_url_hashes(file_list)

for filename in tqdm(hashs):
  shutil.copy(
    os.path.join('/home/xuyang/data/articles',filename+'.story'),
    os.path.join('../data/cnn_articles/', filename+'.story')
  )
  shutil.copy(
    os.path.join('/home/xuyang/data/abstracts',filename+'.story'),
    os.path.join('../data/cnn_abstracts/', filename+'.story')
  )
  shutil.copy(
    os.path.join('/home/xuyang/data/headline_lda',filename+'.html'),
    os.path.join('../data/cnn_head_lda/', filename+'.story')
  )

# lists = read_text_file('url_lists/cnn_wayback_test_urls.txt')
