from bs4 import BeautifulSoup
import os
from tqdm import tqdm

file_list = os.listdir('./downloads')
path_to_story = './processed'
if not os.path.exists(path_to_story):
    os.mkdir(path_to_story)

for filename in tqdm(file_list):

    with open(os.path.join('./downloads',filename),'rb') as f:
        try:
            soup = BeautifulSoup(f.read(),'lxml')
            headline = soup.select('div.article-text > h1')
            if len(headline) == 0:
                headline = soup.select('head > title')
            headline = headline[0].text + '\n'
        except Exception as e:
            print(e)
            print('filename',filename)
            raise Exception('filename error')
        with open(os.path.join(path_to_story,filename),'w') as fou:
            fou.write(headline)
        # print(soup.select('div.article-text > h1'))