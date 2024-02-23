#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[13]:


path = os.getcwd()
path


# In[14]:


files_path = os.chdir("C:/Users/Pallavi/files")
files_path


# In[16]:


student_files = [doc for doc in os.listdir(files_path) if doc.endswith('.txt')]
student_notes = [open(_file, encoding='utf-8').read()
                 for _file in student_files]


# In[17]:


student_files


# In[18]:


student_notes


# In[19]:


def vectorize(Text): 
    return TfidfVectorizer().fit_transform(Text).toarray()
def similarity(doc1, doc2): 
    return cosine_similarity([doc1, doc2])


# In[21]:


vectors = vectorize(student_notes)
s_vectors = list(zip(student_files, vectors))
plagiarism_results = set()
print(vectors)


# In[22]:


s_vectors


# In[23]:


def check_plagiarism():
    global s_vectors
    for student_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((student_a, text_vector_a))
        del new_vectors[current_index]
        for student_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            student_pair = sorted((student_a, student_b))
            score = (student_pair[0], student_pair[1], sim_score)
            plagiarism_results.add(score)
    return plagiarism_results


# In[24]:


for data in check_plagiarism():
    print(data)


# In[ ]:




