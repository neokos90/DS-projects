#!/usr/bin/env python
# coding: utf-8

# # Задание
# 
# Вас пригласили поучаствовать в одном из проектов UNICEF — международного подразделения ООН, чья миссия состоит в повышении уровня благополучия детей по всему миру. 
# 
# Суть проекта — отследить влияние условий жизни учащихся в возрасте от 15 до 22 лет на их успеваемость по математике, чтобы на ранней стадии выявлять студентов, находящихся в группе риска.
# 
# И сделать это можно с помощью модели, которая предсказывала бы результаты госэкзамена по математике для каждого ученика школы (вот она, сила ML!). Чтобы определиться с параметрами будущей модели, проведите разведывательный анализ данных и составьте отчёт по его результатам. 
# 

# # 1. Пункт
# 
# Проведите первичную обработку данных. Так как данных много, стоит написать функции, которые можно применять к столбцам определённого типа. 
# 
# Посмотрим, какие данные есть в нашем датасете.

# In[75]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

df = pd.read_csv('stud_math.xls')

with pd.option_context('display.max_columns', None):
    display(df.info())
    display(df.sample(10))


# **Выводы:** 
# 
# 1. Перевод признаков из одного типа данных в другой не требуется, так как все категориальные признаки относятся к типу "object", а количественные - к числовым типам данных. 
# 2. Признак "studytime, granular" в документации не описан, неизвестно, какие условия жизни он характеризует, так что удалим его из датасета.
# 3. Заменим во всех столбцах пустые значения на None
# 4. Удалим из датасета строки, в которых целевой признак "score" равен NaN.
# 

# In[76]:


# Удаляем столбец "studytime, granular"
df.drop(['studytime, granular'], inplace = True, axis = 1)


# In[77]:


display(df.info())


# In[78]:


# Удалим из датасета строки, в которых целевой признак "score" равен NaN

df.dropna(subset = ['score'], inplace = True)
df.info()


# In[79]:


# Заменим во всех столбцах пустые значения на None

columns = df.columns

def is_digit(element):
    ''' Функция проверяет, что параметр является числом '''
    if str(element).isdigit():
        return True
    else:
        try:
            float(element)
            return True
        except ValueError:
            return False
        
        
# Первичная обработка данных, заменяем пустые значения и NaN на None
for column in columns:
    if df.loc[:, column].dtype == 'object': 
        df.loc[:, column].astype(str).apply(lambda x: None if x.strip() == '' else None if pd.isnull(x) else x)
    elif (df.loc[:, column].dtype == 'float64') or (df.loc[:, column].dtype == 'int64'):
        df.loc[:, column].apply(lambda x: x if is_digit(x) else None )


# # 2. Пункт
# 
# Посмотрите на распределение признака для числовых переменных, устраните выбросы.

# ## score

# In[80]:


# Рассмотрим столбец "score" - это целевой признак, именно его будущая модель будет учиться предсказывать. 
# Посмотрим на распределение оценок:

df.score.hist()
df.score.describe()


# Как мы видим, шкала оценки результатов госэкзамена по математике изменятся от 0 до 100. 
# 
# Большинство оценок распределены от 40 до 70.
# 
# Выбросов нет.

# ## age - возраст

# In[81]:


with pd.option_context('display.max_rows', None):
    display(df.age)


# In[82]:


# Посмотрим на распределение оценок:

df.age.hist(bins = 8)
df.age.describe()


# Большая часть учеников находится в возрасте от 16 до 18 лет.
# 
# В целом в школе обучаются ученики в возрасте от 15 до 22 лет. При этом количество учеников, возраст которых превышает 20 лет крайне мало.

# ## Medu - образование матери (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[83]:


# Посмотрим на распределение оценок:

df.Medu.astype(str).hist(bins = 5)
print(df.Medu.describe())
df.Medu.astype(str).describe()


# Три пропущенных значения заменим на значение медианы.

# In[84]:


median = df['Medu'].median()
df['Medu'] = df['Medu'].apply(lambda x: median if pd.isnull(x) else x)


# ## Fedu - образование отца (0 - нет, 1 - 4 класса, 2 - 5-9 классы, 3 - среднее специальное или 11 классов, 4 - высшее)

# In[85]:


# Посмотрим на распределение оценок:

df.Fedu.astype(str).hist(bins = 7)
print(df.Fedu.describe())
df.Fedu.astype(str).describe()


# В значениях имеются выбросы, нужно от них избавиться.

# In[86]:


df = df.drop(df[df['Fedu'] > 4].index)


# In[87]:


# Заменим пропущенное значение средним

mean = df['Fedu'].mean()
df['Fedu'] = df['Fedu'].apply(lambda x: mean if pd.isnull(x) else x)


# ## traveltime - время в пути до школы (1 - <15 мин., 2 - 15-30 мин., 3 - 30-60 мин., 4 - >60 мин.)

# In[88]:


# Посмотрим на распределение оценок:

df.traveltime.hist()
print(df.traveltime.describe())
print(df.traveltime.astype(str).describe())


# In[89]:


# Выбросов нет 
# Заменим пропущенные значения наиболее часто встречающимся, медианой

median = df['traveltime'].median()
df['traveltime'] = df['traveltime'].apply(lambda x: median if pd.isnull(x) else x)


# ## studytime - время на учёбу помимо школы в неделю (1 - <2 часов, 2 - 2-5 часов, 3 - 5-10 часов, 4 - >10 часов)

# In[90]:


# Посмотрим на распределение оценок:

df.studytime.hist()
df.studytime.describe()


# In[91]:


# Выбросов нет 
# Заменим пропущенные значения наиболее часто встречающимся, медианой

median = df['studytime'].median()
df['studytime'] = df['studytime'].apply(lambda x: median if pd.isnull(x) else x)


# ## failures — количество внеучебных неудач (n, если 1<=n<=3, иначе 0)

# In[92]:


# Посмотрим на распределение оценок:

df.failures.astype(str).hist()
df.failures.describe()


# In[93]:


# Выбросов нет
# Пропущенные значение необходимо заменить на 0

df['failures'] = df['failures'].apply(lambda x: 0 if pd.isnull(x) else x)


# ## famrel — семейные отношения (от 1 - очень плохо до 5 - очень хорошо)

# In[94]:


# Посмотрим на распределение оценок:

df.famrel.astype(str).hist()
df.famrel.describe()


# In[95]:


# Есть выбросы и пустые значения, заменим их на значение медианы

median = df['famrel'].median()
df['famrel'] = df['famrel'].apply(lambda x: median if pd.isnull(x) else median if x == -1 else x)


# ## freetime — свободное время после школы (от 1 - очень мало до 5 - очень мого)

# In[96]:


# Посмотрим на распределение оценок:

df.freetime.astype(str).hist()
df.freetime.describe()


# In[97]:


# Выбросов нет, пустые значения заменим значением медианы

median = df['freetime'].median()
df['freetime'] = df['freetime'].apply(lambda x: median if pd.isnull(x) else x)
df.freetime.describe()


# ## goout — проведение времени с друзьями (от 1 - очень мало до 5 - очень много)

# In[98]:


# Посмотрим на распределение оценок:

df.goout.astype(str).hist()
df.goout.describe()


# In[99]:


# Выбросов нет, пустые значения заменим значением медианы

median = df['goout'].median()
df['goout'] = df['goout'].apply(lambda x: median if pd.isnull(x) else x)
df.goout.describe()


# ## health — текущее состояние здоровья (от 1 - очень плохо до 5 - очень хорошо)

# In[100]:


# Посмотрим на распределение оценок:

df.health.astype(str).hist()
df.health.describe()


# In[101]:


# Выбросов нет, пустые значения заменим значением медианы, хотя здесь можно поэкспериментировать с двумя значениями 3 и 4

median = df['health'].median()
df['health'] = df['health'].apply(lambda x: median if pd.isnull(x) else x)
df.health.describe()


# ## absences — количество пропущенных занятий

# In[102]:


# Посмотрим на распределение оценок:

df.absences.hist()
df.absences.describe()


# In[103]:


# Здесь явно есть выбросы, но возможно, это реальные цифры, когда школьник пропускает занятия по болезни 
# или не уважительной причине 
# Пустые значения заменим медианой

median = df['absences'].median()
df['absences'] = df['absences'].apply(lambda x: median if pd.isnull(x) else x)
df.absences.describe()


# # 3. Пункт
# 
# Оцените количество уникальных значений для номинативных переменных.

# ## school — аббревиатура школы, в которой учится ученик

# In[104]:


df.school.value_counts()


# In[105]:


# Большая часть учеников, данные о которых представлены в датасете, обучаются в школе GP
# Пустых значений нет


# ## sex — пол ученика ('F' - женский, 'M' - мужской)

# In[106]:


print(df.sex.value_counts())
df.loc[:, 'sex'].describe()


# In[107]:


# Мы имеем практически одинаковое соотношение мальчиков и девочек
# Пустых значений нет


# ## address — тип адреса ученика ('U' - городской, 'R' - за городом)

# In[108]:


print(df.address.value_counts())
df.loc[:, 'address'].describe()


# In[109]:


# Оставим пустые значения не заполненными


# ## famsize — размер семьи('LE3' <= 3, 'GT3' >3)

# In[110]:


print(df.famsize.value_counts())
df.loc[:, 'famsize'].describe()


# In[111]:


# Оставим пустые значения не заполненными


# ## Pstatus — статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)

# In[112]:


print(df.Pstatus.value_counts())
df.loc[:, 'Pstatus'].describe()


# In[113]:


# Оставим пустые значения не заполненными


# ## Mjob — работа матери ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)

# In[114]:


print(df.Mjob.value_counts())
df.loc[:, 'Mjob'].describe()


# In[115]:


# Оставим пустые значения не заполненными


# ## Fjob — работа отца ('teacher' - учитель, 'health' - сфера здравоохранения, 'services' - гос служба, 'at_home' - не работает, 'other' - другое)

# In[116]:


print(df.Fjob.value_counts())
df.loc[:, 'Fjob'].describe()


# In[117]:


# Оставим пустые значения не заполненными


# ## reason — причина выбора школы ('home' - близость к дому, 'reputation' - репутация школы, 'course' - образовательная программа, 'other' - другое)

# In[118]:


print(df.reason.value_counts())
df.loc[:, 'reason'].describe()


# In[119]:


# Оставим пустые значения не заполненными


# ## guardian — опекун ('mother' - мать, 'father' - отец, 'other' - другое)

# In[120]:


print(df.guardian.value_counts())
df.loc[:, 'guardian'].describe()


# In[121]:


# Оставим пустые значения не заполненными


# ## schoolsup — дополнительная образовательная поддержка (yes или no)

# In[122]:


print(df.schoolsup.value_counts())
df.loc[:, 'schoolsup'].describe()


# In[123]:


# Оставим пустые значения не заполненными


# ## famsup — семейная образовательная поддержка (yes или no)

# In[124]:


print(df.famsup.value_counts())
df.loc[:, 'famsup'].describe()


# In[125]:


# Оставим пустые значения не заполненными


# ## paid — дополнительные платные занятия по математике (yes или no)

# In[126]:


print(df.paid.value_counts())
df.loc[:, 'paid'].describe()


# In[127]:


# Оставим пустые значения не заполненными


# ## activities — дополнительные внеучебные занятия (yes или no)

# In[128]:


print(df.activities.value_counts())
df.loc[:, 'activities'].describe()


# In[129]:


# Оставим пустые значения не заполненными


# ## nursery — посещал детский сад (yes или no)

# In[130]:


print(df.nursery.value_counts())
df.loc[:, 'nursery'].describe()


# In[131]:


# Оставим пустые значения не заполненными


# ## higher — хочет получить высшее образование (yes или no)

# In[132]:


print(df.higher.value_counts())
df.loc[:, 'higher'].describe()


# In[133]:


# Оставим пустые значения не заполненными


# ##  internet — наличие интернета дома (yes или no)

# In[134]:


print(df.internet.value_counts())
df.loc[:, 'internet'].describe()


# In[135]:


# Оставим пустые значения не заполненными


# ## romantic — в романтических отношениях (yes или no)

# In[136]:


print(df.romantic.value_counts())
df.loc[:, 'romantic'].describe()


# In[137]:


# Оставим пустые значения не заполненными


# # Пункт 3.
# 
# Проведите корреляционный анализ количественных переменных

# In[138]:


# Посмотрим на корреляции количественных признаков.

columns = df.columns
quantitative_signs = []

for column in columns:
    if df.loc[:, column].dtype != 'object':
        quantitative_signs.append(column)
        
corr_matrix = df[quantitative_signs].corr()
sns.heatmap(corr_matrix)


# По раскрашеной матрице корреляций видно, что наиболее высокий коэффициент корреляции имеют такие признаки, как Fedu и Мedu, а также freetime и goout. Из первой связи можно сделать вывод о том, что отец и мать каждого ученика имеют примерно один уровень образования. Из второй связи можно сделать вывод, что свободное время ученики нередко проводят с друзьями.
# 
# Одни из самых низких отрицательных коэффициентов корреляции имеют такие признаки как Fedu и failures, Medu и failures, score и failures, что может говорить о том, что уровень образования влияет на количество неудач ученика, а количество неудач в свою очередь влияет на сдачу экзамена.

# In[139]:


df[quantitative_signs].corr()


# Отнесем к некоррелирующим переменным все, коэффициент корреляции которых находится в промежутку от 0 до 0.1 по модулю.
# (Как это делать правильно, я не нашла)
# 
# Посмотрим на распределение интересующих нас количественных признаков.

# In[140]:


df[quantitative_signs].hist(figsize=(20,12))


# Видим, что все признаки имеют распределение со смещением.
# 
# Построим pairplot, где на главной диагонали рисуются распределения признаков, а вне главной диагонали – 
# диаграммы рассеяния для пар признаков.

# In[141]:


sns.pairplot(df[quantitative_signs])


# # Пункт 4. 
# 
# Теперь рассмотрим номинативные признаки.

# In[142]:


columns = df.columns
nominative_signs = []

for column in columns:
    if df.loc[:, column].dtype == 'object':
        nominative_signs.append(column)
        
# nominative_signs = list(set(nominative_signs) - set(['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
#                                                      'internet', 'romantic']))
print(nominative_signs)


# In[143]:


def get_boxplot(column):
    fig, ax = plt.subplots(figsize = (8, 4))
    sns.boxplot(x=column, y='score', data=df, ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


# In[144]:


for col in nominative_signs:
    get_boxplot(col)


# По графикам похоже, что все параметры, кроме может быть reason, famsup, paid, romantic, могут влиять на итоговый балл по математике.
# 
# Однако графики являются лишь вспомогательным инструментом, настоящую значимость различий может помочь распознать статистика. Проверим, есть ли статистическая разница в распределении оценок по номинативным признакам, с помощью **теста Стьюдента**. Проверим нулевую гипотезу о том, что распределения оценок батончиков по различным параметрам неразличимы:

# In[159]:


def get_stat_dif(column):
    cols = df.loc[:, column].value_counts().index[:]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(df.loc[df.loc[:, column] == comb[0], 'score'], 
                        df.loc[df.loc[:, column] == comb[1], 'score']).pvalue \
            <= 0.05/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


# In[160]:


for col in nominative_signs:
    get_stat_dif(col)


# Как мы видим, серьёзно отличаются 6 параметров: sex,  address, Mjob, higher и romantic. Оставим эти переменные в датасете для дальнейшего построения модели. 
# 
# Итак, в нашем случае важные переменные, которые, возможно, оказывают влияние на оценку, это: sex,  address, Mjob, higher, romantic, failures.

# # Выводы:
# 
# Итак, в результате EDA для анализа влияния параметров условий жизни учащихся на успеваемость по математике были получены следующие выводы:
# 
# - В данных достаточно мало пустых значений.
# - Выбросы найдены только в столбцах с Fedu и famrel, что позволяет сделать вывод о том, что данные достаточно чистые.
# - Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это sex,  address, Mjob, higher, romantic, failures.

# In[ ]:




