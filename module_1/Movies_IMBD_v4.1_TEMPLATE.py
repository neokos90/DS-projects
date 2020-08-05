#!/usr/bin/env python
# coding: utf-8

# In[193]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations


# In[194]:


data = pd.read_csv('movie_bd_v5.xls')
data.sample(5)


# In[195]:


data.info()


# # Предобработка

# In[196]:


answers = {}

# Preprocessing columns
# changing type columns 'release_date' from 'object' to 'datetime'
data['release_date'] = pd.to_datetime(data['release_date'])

# adding extra column 'profit' is equal to the difference revenue minus budget
profit = data['revenue'] - data['budget']
data.insert(loc = 3, column = 'profit', value = profit)


def preprocess_list(some_list):
    """ Function create and return new_list. If value of some_list is list split this element """
    new_list = []    
    for item in some_list:
        if ('|' in item):
            tmp = item.split('|')
            for sub_item in tmp:
                new_list.append(sub_item)
        else: 
            new_list.append(item)
            
    return new_list


# # 1. У какого фильма из списка самый большой бюджет?

# Использовать варианты ответов в коде решения запрещено.    
# Вы думаете и в жизни у вас будут варианты ответов?)

# In[197]:


# +
answers['1'] = '1. Pirates of the Caribbean: On Stranger Tides (tt1298650)'


# In[198]:


data.loc[data['budget'] == data['budget'].max()]


# # 2. Какой из фильмов самый длительный (в минутах)?

# In[199]:


# +
answers['2'] = '2. Gods and Generals (tt0279111)'


# In[200]:


data.loc[data.runtime == data.runtime.max()]


# # 3. Какой из фильмов самый короткий (в минутах)?
# 
# 
# 
# 

# In[201]:


# +
answers['3'] = '3. Winnie the Pooh (tt1449283)'


# In[202]:


data.loc[data.runtime == data.runtime.min()]


# # 4. Какова средняя длительность фильмов?
# 

# In[203]:


# +
answers['4'] = '4. 110'


# In[204]:


round(data.runtime.mean())


# # 5. Каково медианное значение длительности фильмов? 

# In[205]:


# +
answers['5'] = '5. 107'


# In[206]:


data.runtime.median()


# # 6. Какой самый прибыльный фильм?
# #### Внимание! Здесь и далее под «прибылью» или «убытками» понимается разность между сборами и бюджетом фильма. (прибыль = сборы - бюджет) в нашем датасете это будет (profit = revenue - budget) 

# In[207]:


# +
answers['6'] = '6. Avatar (tt0499549)'


# In[208]:


data.loc[data.profit == data.profit.max()]


# # 7. Какой фильм самый убыточный? 

# In[209]:


# +
answers['7'] = '7. The Lone Ranger (tt1210819)'


# In[210]:


data.loc[data.profit == data.profit.min()]


# # 8. У скольких фильмов из датасета объем сборов оказался выше бюджета?

# In[211]:


# +
answers['8'] = '8. 1478'


# In[212]:


len(data.query('revenue > budget'))


# ВАРИАНТ 2

# In[213]:


len(data.loc[data.revenue > data.budget])


# # 9. Какой фильм оказался самым кассовым в 2008 году?

# In[214]:


# +
answers['9'] = '9. The Dark Knight (tt0468569)'


# In[215]:


max_profit = data.groupby('release_year')['profit'].max()
data.loc[data.profit == max_profit[2008]]


# In[216]:


data.loc[data.profit == data.query('release_year == "2008"').profit.max()]


# # 10. Самый убыточный фильм за период с 2012 по 2014 г. (включительно)?
# 

# In[217]:


# +
answers['10'] = '10. The Lone Ranger (tt1210819)'


# In[218]:


min_profit = data[data['release_year'].isin([2012, 2013, 2014])].profit.min()
data.loc[data.profit == min_profit]


# ВАРИАНТ 2

# In[219]:


min_profit = data.query('release_year in [2012, 2013, 2014]').profit.min()
data.loc[data.profit == min_profit]


# # 11. Какого жанра фильмов больше всего?

# In[220]:


# +
answers['11'] = '11. Drama'


# In[221]:


answer = Counter(preprocess_list(data['genres'].tolist())).most_common(1)
answer


# # 12. Фильмы какого жанра чаще всего становятся прибыльными? 

# In[222]:


# +
answers['12'] = '12. Drama'


# In[223]:


answer = Counter(preprocess_list(data.loc[data.profit > 0]['genres'].tolist())).most_common(1)
answer


# # 13. У какого режиссера самые большие суммарные кассовые сбооры?

# In[224]:


# +
answers['13'] = '13. Peter Jackson'
data.groupby(['director'])['profit'].sum().sort_values(ascending = False)


# # 14. Какой режисер снял больше всего фильмов в стиле Action?

# In[225]:


# +
answers['14'] = '14. Robert Rodriguez'


# In[226]:


df_group = data.loc[data['genres'].str.contains('Action', na = False)].director
directors_list = []
for item in df_group:
    if('|' in item):
        tmp = item.split('|')
        for sub_item in tmp:
            directors_list.append(sub_item)
    else:
        directors_list.append(item)

Counter(directors_list).most_common(1)


# # 15. Фильмы с каким актером принесли самые высокие кассовые сборы в 2012 году? 

# In[227]:


# +
answers['15'] = '15. Chris Hemsworth'


# In[228]:


data[data.profit == data.loc[data.release_year == 2012]['profit'].max()].cast.tolist()


# # 16. Какой актер снялся в большем количестве высокобюджетных фильмов?

# In[229]:


# +
answers['16'] = '16. Matt Damon'


# In[230]:


answer = Counter(preprocess_list(data.loc[data.budget > data.budget.mean()]['cast'].tolist())).most_common(1)
print(answer)


# # 17. В фильмах какого жанра больше всего снимался Nicolas Cage? 

# In[231]:


# +
answers['17'] = '17. Action'


# In[232]:


answer = Counter(preprocess_list(data.loc[data.cast.str.contains('Nicolas Cage', na = False)]['genres'].tolist())).most_common(1)
print(answer)


# # 18. Самый убыточный фильм от Paramount Pictures

# In[233]:


# +
answers['18'] = '18. K-19: The Widowmaker (tt0267626)'


# In[234]:


min_profit = data.loc[data['production_companies'].str.contains('Paramount Pictures', na = False)]['profit'].min()
data.loc[data.profit == min_profit]


# # 19. Какой год стал самым успешным по суммарным кассовым сборам?

# In[235]:


# +
answers['19'] = '19. 2015'


# In[236]:


data.groupby('release_year')['revenue'].sum().sort_values(ascending = False).idxmax()


# # 20. Какой самый прибыльный год для студии Warner Bros?

# In[237]:


# +
answers['20'] = '20. 2014'


# In[238]:


(data.loc[data['production_companies'].str.contains('Warner Bros', na = False)]
 .groupby('release_year')['profit'].sum().sort_values(ascending = False)).idxmax()


# # 21. В каком месяце за все годы суммарно вышло больше всего фильмов?

# In[239]:


# +
answers['21'] = '21. 9'


# In[240]:


data['release_date'].apply(lambda x: x.strftime("%B")).value_counts().idxmax()


# ВАРИАНТ 2

# In[241]:


data['release_date'].apply(lambda x: x.month).value_counts().idxmax()


# # 22. Сколько суммарно вышло фильмов летом? (за июнь, июль, август)

# In[242]:


# +
answers['22'] = '22. 450'


# In[243]:


films = data['release_date'].apply(lambda x: x.month).value_counts()
answer = films[6] + films[7] + films[8]
print(answer)


# # 23. Для какого режиссера зима – самое продуктивное время года? 

# In[244]:


# +
answers['23'] = '23. Peter Jackson'


# In[245]:


# Для ответа на вопрос понадобится информация о режиссерах и дате выхода фильма
df = data[['director', 'release_date']]
tmp_list = []

# Т.к. столбец 'director' может содержать список режиссеров, то нужно распарсить значения 
# Получаем список, где одна строка содержит информацию об одном режиссере и месеце выхода фильма
for index, row in df.iterrows():
    directors_list = row['director']
    if ('|' in directors_list):         
        tmp = directors_list.split('|')
        for sub_item in tmp:
            tmp_list.append([sub_item, row['release_date'].month])
    else: 
        tmp_list.append([row['director'], row['release_date'].month])

# Создаем DataFrame и заполняем его подготовленным списком tmp_list,
# для того чтобы воспрользоваться механизмами группировки DataFrame-ов
new_df = pd.DataFrame(tmp_list, columns = ['director', 'release_month'])
new_df.query('release_month in [12, 1, 2]').groupby('director')['release_month'].count().idxmax()


# # 24. Какая студия дает самые длинные названия своим фильмам по количеству символов?

# In[246]:


# +
answers['24'] = '24. Four By Two Productions'


# In[247]:


# Для ответа на вопрос понадобится информация о названии фильма и о студиях
df = data[['original_title', 'production_companies']]
tmp_list = []

# Т.к. столбец 'production_companies' может содержать список студий, то нужно распарсить значения 
# Получаем список, где одна строка содержит информацию об одной студии и об одном фильме, который она сняла
for index, row in df.iterrows():
    if ('|' in row['production_companies']):
        tmp = row['production_companies'].split('|')
        for company in tmp:
            tmp_list.append([row['original_title'], company, len(row['original_title'])])
    else:
        tmp_list.append([row['original_title'], row['production_companies'], len(row['original_title'])]) 

# Создаем DataFrame и заполняем его подготовленным списком tmp_list,
# для того чтобы воспрользоваться механизмами группировки DataFrame-ов
new_df = pd.DataFrame(tmp_list, columns = ['original_title', 'production_company', 'number_characters'])
new_df.groupby('production_company')['number_characters'].mean().idxmax()


# # 25. Описание фильмов какой студии в среднем самые длинные по количеству слов?

# In[248]:


# +
answers['25'] = '25. Midnight Picture Show'


# In[249]:


# Для ответа на вопрос понадобится информация о студиях и описании фильма
df = data[['overview', 'production_companies']]
tmp_list = []
    
# Т.к. столбец 'production_companies' может содержать список студий, то нужно распарсить значения 
# Получаем список, где одна строка содержит информацию об одной студии и об одном описании фильма, который она сняла
for index, row in df.iterrows():
    number_words = len(row['overview'].split(' '))
    if ('|' in row['production_companies']):
        tmp = row['production_companies'].split('|')
        for company in tmp:
            tmp_list.append([company, number_words])
    else:
        tmp_list.append([row['production_companies'], number_words]) 

# Создаем DataFrame и заполняем его подготовленным списком tmp_list,
# для того чтобы воспрользоваться механизмами группировки DataFrame-ов
new_df = pd.DataFrame(tmp_list, columns = ['production_company', 'number_words'])
new_df.groupby('production_company')['number_words'].mean().idxmax()


# # 26. Какие фильмы входят в 1 процент лучших по рейтингу? 
# по vote_average

# In[250]:


# +
answers['26'] = '26. Inside Out, The Dark Knight, 12 Years a Slave'


# In[251]:


# Рассчитаем какое количество фильмов попадает под 1%
number_films = int(round((data['original_title'].count() * 0.01)))
data[['original_title', 'vote_average']].sort_values(by = 'vote_average', ascending = False).head(number_films)


# # 27. Какие актеры чаще всего снимаются в одном фильме вместе?
# 

# In[252]:


# +
answers['27'] = '27. Daniel Radcliffe & Rupert Grint'


# In[253]:


# Нам понадобятся об актерском составе для каждого фильма
actors_groups = data['cast'].tolist()
actors = []

# Из каждой группы актёров, снимающихся в одном фильме, создаем массив кортежей, вида (actor1, actor2), 
# так чтобы присутствовали все вариации "актёр в паре с другим актёром", но при этом не повторялись
for actor_group in actors_groups:
    if ('|' in actor_group):
        # Т.к. кортежи с одними и теми же значениями, указанными в разном порядке, не равны
        # ('hello', 'world') != ('world', 'hello'), необходимо исключить такие ситуации 
        # для этого используем сортировку
        actor_list = sorted(actor_group.split('|'))
        actor_tuple = list(combinations(actor_list, 2))
        actors = actors + actor_tuple
    else:
        actor_tuple = tuple([actor_group, actor_group])
        actors.append(actor_tuple)
        
# Считаем, сколько раз встречается каждая из пар        
Counter(actors).most_common(10)


# # Submission

# In[254]:


# в конце можно посмотреть свои ответы к каждому вопросу
answers


# In[255]:


# и убедиться что ни чего не пропустил)
len(answers)


# In[ ]:





# In[ ]:




