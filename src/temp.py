#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stanfordcorenlp import StanfordCoreNLP
from mxnet import nd
nlp = StanfordCoreNLP(r'D:\ProgramData\stanford-corenlp-full-2018-10-05')


# In[8]:


class TNode(object):
    def __init__(self):
        self.next = []
        self.val = None


# In[27]:


print(nlp.parse('I am happy.'))


# In[28]:


print(nlp.parse('I am happy'))


# In[4]:


out = nlp.parse('I am happy').replace('\r\n','')


# In[5]:


# sub part count
def count_sub(sub:str):
    p2i = {'(':1,')':-1}
    count = 0
    split_ind = []
    num_of_left = 0
    for k,i in enumerate(sub):
        if i in p2i:
            num_of_left += p2i[i]
            if num_of_left == 0:
                count += 1
                split_ind.append(k+1)
    return count, split_ind


# In[21]:


out


# In[22]:


def parse(dep:str):
    root = TNode()
#     print(dep)
    t = [x.strip() for x in dep[1:-1].split(' ',1)]
    root.value = t[0]
    print(t[0],t[1])
    sub_num, sub_split_ind = count_sub(t[1])
    pre = 0
    for i in range(sub_num):
        root.next.append(parse(t[1][pre:sub_split_ind[i]].strip()))
        pre = sub_split_ind[i]
    return root


# In[24]:


parse(out)


# ### 两个需求
#
# 1. 对单个词的修饰：
#  直接concat
# 2. 多个词的合并：
#  直接加恐怕不合适，用一个Dense +
#
# 句子间使用RNN

# In[14]:


count_sub(out)


# In[15]:


count_sub('(NP (PRP I))\r\n    (VP (VBP am)\r\n      (ADJP (JJ happy)))')


# In[ ]:




