import hashlib
import os

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

for c in cate:
    for d in data_type:
        path = '_'.join([c,'wayback',d,'urls'])+'.txt'
        file_list = read_text_file(path)
        
        print(path)
# lists = read_text_file('url_lists/cnn_wayback_test_urls.txt')
