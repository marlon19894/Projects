
# IBM HR Analytics


The following dataset was created by IBM data scientists and includes 1470 listings. The challenge is to uncover the factors that lead to employee attrition and to create a model that predicts the churn of employees.





Further explanation of different columns:



Education
1 'Below College'
2 'College'
3 'Bachelor'
4 'Master'
5 'Doctor'

EnvironmentSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobInvolvement
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

JobSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

PerformanceRating
1 'Low'
2 'Good'
3 'Excellent'
4 'Outstanding'

RelationshipSatisfaction
1 'Low'
2 'Medium'
3 'High'
4 'Very High'

WorkLifeBalance
1 'Bad'
2 'Good'
3 'Better'
4 'Best'

# 1. First View on the Data


```python
#Standard Imports

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
%matplotlib inline

# Plotly Imports 
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#Import Models for Prediction
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, classification_report)


from imblearn.over_sampling import SMOTE


import xgboost

# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')
```

    /anaconda3/lib/python3.7/site-packages/sklearn/externals/six.py:31: FutureWarning:
    
    The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
    
    /anaconda3/lib/python3.7/site-packages/sklearn/utils/deprecation.py:144: FutureWarning:
    
    The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
    
    /anaconda3/lib/python3.7/site-packages/sklearn/externals/joblib/__init__.py:15: FutureWarning:
    
    sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.
    



```python
#Loading the data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
```


```python
#Checking the data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>DailyRate</th>
      <th>Department</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EducationField</th>
      <th>EmployeeCount</th>
      <th>EmployeeNumber</th>
      <th>...</th>
      <th>RelationshipSatisfaction</th>
      <th>StandardHours</th>
      <th>StockOptionLevel</th>
      <th>TotalWorkingYears</th>
      <th>TrainingTimesLastYear</th>
      <th>WorkLifeBalance</th>
      <th>YearsAtCompany</th>
      <th>YearsInCurrentRole</th>
      <th>YearsSinceLastPromotion</th>
      <th>YearsWithCurrManager</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1102</td>
      <td>Sales</td>
      <td>1</td>
      <td>2</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>279</td>
      <td>Research &amp; Development</td>
      <td>8</td>
      <td>1</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>10</td>
      <td>3</td>
      <td>3</td>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>1373</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>2</td>
      <td>Other</td>
      <td>1</td>
      <td>4</td>
      <td>...</td>
      <td>2</td>
      <td>80</td>
      <td>0</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>1392</td>
      <td>Research &amp; Development</td>
      <td>3</td>
      <td>4</td>
      <td>Life Sciences</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>3</td>
      <td>80</td>
      <td>0</td>
      <td>8</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>7</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>591</td>
      <td>Research &amp; Development</td>
      <td>2</td>
      <td>1</td>
      <td>Medical</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>4</td>
      <td>80</td>
      <td>1</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
#Checking if any data is missing
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
    Age                         1470 non-null int64
    Attrition                   1470 non-null object
    BusinessTravel              1470 non-null object
    DailyRate                   1470 non-null int64
    Department                  1470 non-null object
    DistanceFromHome            1470 non-null int64
    Education                   1470 non-null int64
    EducationField              1470 non-null object
    EmployeeCount               1470 non-null int64
    EmployeeNumber              1470 non-null int64
    EnvironmentSatisfaction     1470 non-null int64
    Gender                      1470 non-null object
    HourlyRate                  1470 non-null int64
    JobInvolvement              1470 non-null int64
    JobLevel                    1470 non-null int64
    JobRole                     1470 non-null object
    JobSatisfaction             1470 non-null int64
    MaritalStatus               1470 non-null object
    MonthlyIncome               1470 non-null int64
    MonthlyRate                 1470 non-null int64
    NumCompaniesWorked          1470 non-null int64
    Over18                      1470 non-null object
    OverTime                    1470 non-null object
    PercentSalaryHike           1470 non-null int64
    PerformanceRating           1470 non-null int64
    RelationshipSatisfaction    1470 non-null int64
    StandardHours               1470 non-null int64
    StockOptionLevel            1470 non-null int64
    TotalWorkingYears           1470 non-null int64
    TrainingTimesLastYear       1470 non-null int64
    WorkLifeBalance             1470 non-null int64
    YearsAtCompany              1470 non-null int64
    YearsInCurrentRole          1470 non-null int64
    YearsSinceLastPromotion     1470 non-null int64
    YearsWithCurrManager        1470 non-null int64
    dtypes: int64(26), object(9)
    memory usage: 402.0+ KB



```python
df['Attrition'].value_counts(normalize=True)
```




    No     0.838776
    Yes    0.161224
    Name: Attrition, dtype: float64



The dataset includes 140 observations and 35 features. This data consists of two different datatypes: Categorical (mostly ordinal variables) and numerical.
Our target variable is the "Attrition" column. 1237 (84% of cases) employees did not leave the organization while 237 (16% of cases) did leave the organization making our dataset to be considered imbalanced since more people stay in the organization than they actually leave. 

Luckily there is no missing data. So we can dig deeper into the data by EDA



# 2. Exploratory Data Analysis

## 2.1 General View on our Target 

First of all, let's have a look on our imbalanced target variable. 84% of employees did not quit the organization while 16% did leave the organization. Knowing that we are dealing with an imbalanced dataset will help us to determine the best model for our predictions:


```python
plt.figure(figsize=[8,6])

colors = ['#43e653','#ed1c3c']

g=sns.countplot(df['Attrition'], palette=colors)

total = float(len(df))
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height/total),
            ha="center") 

g.set_title('Attrition Rate (Total)')

```




    Text(0.5,1,'Attrition Rate (Total)')




![png](output_12_1.png)


## 2.2 The Impact of Income towards Attrition

I wonder how much importance does each employee give to the income they earn in the organization. Here we will find out if it is true that money is really everything!

## Questions

- What is the average monthly income by department? Are there any significant differences between individuals who quit and didn't quit?
- Are there significant changes in the level of income by Job Satisfaction? Are individuals with a lower satisfaction getting much less income than the ones who are more satisfied?
- Do employees who quit the organization have a much lower income than people who didn't quit the organization?
- Do employees with a higher performance rating earn more than with a lower performance rating? Is the difference significant by Attrition status?

## Summary

- Income by Departments: Wow! We can see huge differences in each department by attrition status.

- Income by Job Satisfaction: Hmm. It seems the lower the job satisfaction the wider the gap by attrition status in the levels of income.

- Attrition sample population: I would say that most of this sample population has had a salary increase of less than 15% and a monthly income of less than 7,000

- Exhaustion at Work: Over 54% of workers who left the organization worked overtime! Will this be a reason why employees are leaving?

- Differences in the DailyRate: HealthCare Representatives , Sales Representatives , and Research Scientists have the biggest daily rates differences in terms of employees who quit or didn't quit the organization. This might indicate that at least for the these roles, the sample population that left the organization was mainly because of income.

## Average Income by Department


```python
inc_n = df[df['Attrition']=='No']

inc_y = df[df['Attrition']=='Yes']


avg_inc_n = inc_n.groupby(['Department','Attrition'])['MonthlyIncome'].mean().to_frame()
avg_inc_n.reset_index(inplace=True)

avg_inc_y = inc_y.groupby(['Department','Attrition'])['MonthlyIncome'].mean().to_frame()
avg_inc_y.reset_index(inplace=True)


sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16,6))

g = sns.barplot(x='Department', y='MonthlyIncome', hue='Attrition', data=avg_inc_y, color='#ed1c3c', ax=axes[0])
g.set_ylim(0,8000)
g.legend(loc='upper center')
for p in g.patches:
    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.,p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')




f = sns.barplot(x='Department', y='MonthlyIncome', hue='Attrition', data=avg_inc_n, color='#43e653', ax=axes[1])
f.set_ylim(0,8000)
f.legend(['No'],loc='upper center')
for p in f.patches:
    f.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.,p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

fig.suptitle('Average Income by Department & Attrition', size=16)


```




    Text(0.5,0.98,'Average Income by Department & Attrition')




![png](output_20_1.png)


## Satisfaction by Income


```python
inc_n = df[df['Attrition']=='No']
inc_y = df[df['Attrition']=='Yes']


avg_inc_n = inc_n.groupby(['JobSatisfaction','Attrition'])['MonthlyIncome'].median().to_frame()
avg_inc_n.reset_index(inplace=True)


avg_inc_y = inc_y.groupby(['JobSatisfaction','Attrition'])['MonthlyIncome'].median().to_frame()
avg_inc_y.reset_index(inplace=True)

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16,6))


g = sns.barplot(x='MonthlyIncome', y='JobSatisfaction',data=avg_inc_y, color='#ed1c3c', orient='h', ax=axes[0])
g.set_xlim(0,6000)
g.set_xlabel('Median Income')
g.set_ylabel('Level of Job Satisfaction')
for p in g.patches:
    val= '{:,.0f}$'.format(p.get_width())
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    g.annotate(val,(x,y))



f = sns.barplot(x='MonthlyIncome', y='JobSatisfaction',color='#43e653', orient='h',data=avg_inc_n, ax=axes[1])
f.set_xlim(0,6000)
f.set_xlabel('Median Income')
f.set_ylabel('Level of Job Satisfaction')
for p in f.patches:
    val= '{:,.0f}$'.format(p.get_width())
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    f.annotate(val,(x,y))

red_patch = mpatches.Patch(color='r', label='Yes')
green_patch = mpatches.Patch(color='green', label='No')


fig.suptitle('Is Income a Reason to Leave? \n by Attrition', size=16)
fig.legend(handles=[red_patch, green_patch], title='Attrition')
```




    <matplotlib.legend.Legend at 0x1a24e3ff60>




![png](output_22_1.png)


## Income and its Impact on Attrition


```python
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

sns.set(style="darkgrid")
fig=plt.figure(figsize=(12,10))
fig.tight_layout(pad=3.0)

ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, 0:2])
ax3 = plt.subplot(gs[1, 2:4])

colors = ['#ed1c3c','#43e653']

f=sns.stripplot(x=df['PercentSalaryHike'], y=df['MonthlyIncome'], hue=df['Attrition'], 
              ax=ax1, palette=colors, alpha=0.7, jitter=1.5)
f.legend().set_visible(False)

g=sns.violinplot(x=inc_n['MonthlyIncome'], y=inc_n['PerformanceRating'], orient='h',
                 ax=ax2, color='#43e653', inner=None, alpha=0.5)
g.set_xlabel('')


h=sns.violinplot(x=inc_y['MonthlyIncome'], y=inc_y['PerformanceRating'], orient='h',
                 ax=ax3, inner=None, color='#ed1c3c', sharex=ax2, alpha=0.5)
h.set_ylabel('')
h.set_xlabel('')



fig.legend(df['Attrition'].unique(),    
           labels=df['Attrition'].unique(),   
           loc=6,  
           borderaxespad=0.1,  
           title="Attrition" 
           )

fig.legend([g,f,h], 'Atttio')
fig.text(0.47,0.05,'Monthly Income')
fig.suptitle('Income and its Impact \n on Attrition \n', size=16)
```




    Text(0.5,0.98,'Income and its Impact \n on Attrition \n')




![png](output_24_1.png)


## Level of Attrition by Overtime Status


```python
fig, axes = plt.subplots(ncols=2, figsize=(18,9))



def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


colors = ['#43e653','#ed1c3c']


f=axes[0].pie(inc_y.groupby('OverTime').count()['Age'], labels=['No','Yes'],
              autopct=make_autopct(inc_y.groupby('OverTime').count()['Age']),
              colors=colors, shadow=True)

axes[0].title.set_text('In Percent')



                   
g=axes[1].bar(x=['No','Yes'],height=inc_y.groupby('OverTime').count()['Age'], color=colors,linewidth=3, 
              bottom=1, ecolor='white')
axes[1].set_facecolor('white')
axes[1].title.set_text('In Total')




fig.patch.set_facecolor('white')
fig.suptitle('Level of Attrition by Overtime Status', size=16)
```




    Text(0.5,0.98,'Level of Attrition by Overtime Status')




![png](output_26_1.png)


## 2.3 Working Environment

In this section, we will explore everything that is related to the working environment and the structure of the organization.

## Questions

- Job Roles: How many employees in each Job Role?
- Salary by Job Role: What's the average salary by job role?
- Attrition by Job Role: What's the attrition percentage by job role? Which job role has the highest attrition rate? Which has the lowest?
- Years with Current Manager What's the average satisfaction rate by the status of the manager? Are recently hired managers providinga higher job satisfaction to employees?


## Summary

- Number of Employees by Job Role: Sales and Research Scientist are the job positions with the highest number of employees.
- Salary by Job Role: Managers and Research Directors have the highest salary on average.
- Attrition by Job Role: Sales Representatives, HR and Laboratory Technician have the highest attrition rates. This could give us a hint that in these departments we are experiencing certain issues with employees.
- Managers: Employees that are dealing with recently hired managers have a lower satisfaction score than managers that have been there for a longer time.


## Major Job Roles Part I


```python
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in df.JobRole)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=200, background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Major Job Roles in Company \n Part I\n \n', size=25)
plt.show()
```


    <Figure size 720x432 with 0 Axes>



![png](output_34_1.png)


## Major Job Roles Part II 


```python
import squarify
fig=plt.figure(figsize=[10,10])

my_values = df.groupby('JobRole').count()['Age']

cmap = plt.cm.Blues
mini=min(my_values)
maxi=max(my_values)
colors = [cmap(value) for value in my_values]



squarify.plot(sizes=df.groupby('JobRole').count()['Age'], label=df.groupby('JobRole').count().index,alpha=0.8, color=colors)
plt.axis('off')
plt.gca().invert_yaxis()

plt.title("Major Job Roles in Company \n Part II", fontsize=25)



```




    Text(0.5,1,'Major Job Roles in Company \n Part II')




![png](output_36_1.png)


## Median Vs. Mean Income by Job Role


```python
inc_med = df.groupby('JobRole')['MonthlyIncome'].median().sort_values(ascending=False)
inc_mea = df.groupby('JobRole')['MonthlyIncome'].mean().sort_values(ascending=False)



sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(16,6))

g = sns.barplot(x=inc_med.index, y=inc_med, data=df, color='#0071CB', ax=axes[0])
g.set_ylim(0,20000)
g.set_ylabel('Median Income')
g.set_xlabel('')
g.set_xticklabels(inc_med.index,rotation=90)

for p in g.patches:
    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.,p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



f = sns.barplot(x=inc_mea.index, y=inc_mea, data=df, color='#9BD3FF', ax=axes[1])
f.set_ylim(0,20000)
f.set_ylabel('Mean Income')
f.set_xlabel('')
f.set_xticklabels(inc_mea.index,rotation=90)

for p in f.patches:
    f.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2.,p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')



    
fig.suptitle('Median vs. Mean Income by different Job Roles', size=16)


```




    Text(0.5,0.98,'Median vs. Mean Income by different Job Roles')




![png](output_38_1.png)


## Attrition by Job Role


```python
# Data
ct_t = df[['JobRole', 'Attrition', 'Age']].groupby(['JobRole','Attrition'])['Age'].count()
ct_t = pd.DataFrame(ct_t)
ct_t.reset_index(inplace=True)
ct = ct_t.JobRole.unique()
nu = [[0],[1]]

k=[]

for x in ct:
    k.append((ct_t[ct_t['JobRole']==x]['Age'].iloc[0]/(ct_t[ct_t['JobRole']==x]['Age'].iloc[0] + ct_t[ct_t['JobRole']==x]['Age'].iloc[1])).round(2))

k=[0.93, 0.07,0.77,0.23,0.76,0.24,0.95,0.05,0.93,0.07,0.98,0.02,0.84,0.16,0.83,0.17,0.6,0.4]


ct_t['Percentage'] =k
ct_t.drop('Age', 1, inplace=True)





job_r = list(ct_t.JobRole.unique())

att_y = ct_t[ct_t['Attrition']=='Yes']['Percentage'].values

att_n = ct_t[ct_t['Attrition']=='No']['Percentage'].values


# Sort by number of sales staff
idx = att_n.argsort()
job_r, att_y, att_n = [np.take(x, idx) for x in [job_r, att_y, att_n]]
y = np.arange(att_y.size)

fig, axes = plt.subplots(ncols=2, sharey=True, figsize=[10,8])
axes[0].barh(y, att_n, align='center', color='#43e653', zorder=10)
axes[0].set(title='NO')

axes[1].barh(y, att_y, align='center', color='#ed1c3c', zorder=10)
axes[1].set(title='YES')
axes[1].set_xlim(xmax=1)

axes[0].invert_xaxis()
axes[0].set(yticks=y, yticklabels=job_r)
axes[0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.7)


plt.setp(axes[0].yaxis.get_majorticklabels(), ha='center')


dx = 80 / 72.
dy = 0 / 72.
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

for label in axes[0].yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)


fig.suptitle('Attrition by JobRole', size=18)

plt.show()
```


![png](output_40_0.png)


## Current Managers & Average Satisfaction Score


```python
inc_n = df[df['Attrition']=='No']
inc_y = df[df['Attrition']=='Yes']

def year_cat_man(year):
    if year <=1:
        return 'Recently Hired'
    elif 1<year<=4:
        return '2-4 Years Hired'
    else:
        return 'Established Manager'
    

inc_n['Man_Year'] = inc_n['YearsWithCurrManager'].apply(year_cat_man)
inc_y['Man_Year'] = inc_y['YearsWithCurrManager'].apply(year_cat_man) 


inc_n.groupby('Man_Year')['JobSatisfaction'].mean().reset_index()
inc_y.groupby('Man_Year')['JobSatisfaction'].mean().reset_index()


gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

sns.set(style="white")
fig=plt.figure(figsize=(12,10))
fig.tight_layout(pad=3.0)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[1, 0:2])
ax4 = plt.subplot(gs[1, 2:4])




f = sns.barplot(x='JobSatisfaction', y='Man_Year',color='#43e653', orient='h',
                data=inc_n.groupby('Man_Year')['JobSatisfaction'].mean().reset_index(), ax=ax1)
f.set_xlabel('')
f.set_ylabel('Years with current Manager')
for p in f.patches:
    val= '{:,f}'.format(p.get_width())
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    f.annotate(val,(x-0.55,y))

    
    
g = sns.barplot(x='JobSatisfaction', y='Man_Year',color='#ed1c3c', orient='h',
                data=inc_y.groupby('Man_Year')['JobSatisfaction'].mean().reset_index(), ax=ax2)
g.set_xlim(0,3)
g.set_xlabel('')
g.set_ylabel('')
g.set_yticklabels('')
for p in g.patches:
    val= '{:,f}'.format(p.get_width())
    width, height =p.get_width(),p.get_height()
    x=p.get_x()+width+0.02
    y=p.get_y()+height/2
    g.annotate(val,(x-0.55,y))
    

h = sns.kdeplot(data=inc_n['RelationshipSatisfaction'], color='#43e653', ax=ax3, shade=True, label='Attrition = No')
h.set_xlabel('')
h.set_ylabel('Density') 
h.set_ylim(0,0.5)
h.set_xlim(1,4)

i = sns.kdeplot(data=inc_y['RelationshipSatisfaction'], color='#ed1c3c', ax=ax4, shade=True, label='Attrition = Yes')
i.set_xlabel('')
i.set_ylabel('') 
i.set_ylim(0,0.5)
i.set_xlim(1,4)


fig.text(0.455,0.48,'Average Satisfaction')
fig.text(0.455,0.05,'Relationship Satisfaction')
fig.suptitle('Dealing With Current Managers \n', size=16)

```




    Text(0.5,0.98,'Dealing With Current Managers \n')




![png](output_42_1.png)


## 2.4 In-Depth Look Into Attrition

Digging into Attrition:

In this section, we will go as deep as we can into employees that quit to have a better understanding what were some of the reasons that employees decided to leave the organization.

Questions to Ask Ourselves:
- Attrition by Department: How many employees quit by Department? Did they have a proper work-life balance?



```python
gs = gridspec.GridSpec(2, 6)
gs.update(wspace=0.5)

sns.set(style="white")
fig=plt.figure(figsize=(18,12))
fig.tight_layout(pad=2.0)


ax1 = plt.subplot(gs[1, :2])
ax2 = plt.subplot(gs[1, 2:4])
ax3 = plt.subplot(gs[1, 4:6])


f = sns.barplot(x=inc_y[inc_y['Department']=='Human Resources'].groupby('WorkLifeBalance')['Age'].count().index,
                y=inc_y[inc_y['Department']=='Human Resources'].groupby('WorkLifeBalance')['Age'].count(), 
                color='#47A2E9',ax=ax1)
f.set_ylim(0,70)
f.set_ylabel('Number of Employees', fontsize=16)
f.set_xlabel('')
f.set_title('Human Ressources',fontweight="bold", size=14)
for p in f.patches:
    f.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width() / 2.,p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    
    
    
g = sns.barplot(x=inc_y[inc_y['Department']=='Sales'].groupby('WorkLifeBalance')['Age'].count().index,
                y=inc_y[inc_y['Department']=='Sales'].groupby('WorkLifeBalance')['Age'].count(),
                color='#BF0F9F', ax=ax2)
g.set_ylim(0,70)
g.set_ylabel('')
g.set_xlabel('Work & Life Balance', fontsize=16)
g.set_title('Sales',fontweight="bold", size=14)
for p in g.patches:
    g.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width() / 2.,p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
  


    
h = sns.barplot(x=inc_y[inc_y['Department']=='Research & Development'].groupby('WorkLifeBalance')['Age'].count().index,
                y=inc_y[inc_y['Department']=='Research & Development'].groupby('WorkLifeBalance')['Age'].count(), 
                color='#10AD93',ax=ax3)
h.set_ylim(0,70)
h.set_ylabel('')
h.set_xlabel('')
h.set_title('Research & Development',fontweight="bold", size=14)

for p in h.patches:
    h.annotate(format(p.get_height(), ''), (p.get_x() + p.get_width() / 2.,p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    


```


![png](output_46_0.png)


## 2.5 Correlations and Bi-variate Analysis:

Correlation Matrix:

In this section we will understand what features have a positive correlation with each other. This tells us whether there is an association between two variables. What I like about the correlation section is that it gives us a better understanding of some of the features we are dealing with

Summary:
- The higher the total working years the higher the monthly income of an employee.
- The higher the percent salary hike the higher the performance rating.
- The higher the years with current manager the higher the years since last promotion.
- The higher the age the higher the monthly income.

## 2.5.1 Correlations


```python
#Numerical encoding of the target variable

target_map = {'Yes':1, 'No':0}
df['Attrition_numerical']=df['Attrition'].apply(lambda x: target_map[x])
```


```python
#Getting all columns with numerical data for further exploration

df.drop(['EmployeeCount', 'StandardHours' ], 1, inplace=True)

numerical=[]

for a in df.columns:
    if df[a].dtypes=='int64':
        numerical.append(a)
        
```


```python
x = df[numerical]
y = df[numerical]

z = x.corr()
```


```python
#Checking the numerical data for Correlation. Using Plotly to get an interactive plot

data=go.Heatmap(z=z, x=df[numerical].columns.values, y=df[numerical].columns.values, type='heatmap', colorscale = 'Viridis', reversescale = False)

layout = go.Layout(
    title='Pearson Correlation of numerical features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 900)
    
    
fig = go.Figure(data=data, layout=layout)


py.iplot(fig, filename='labelled-heatmap')
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v1.52.2
* Copyright 2012-2020, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



<div>
        
        
            <div id="c6fb06ee-6fb8-409f-a02d-dc7b0e1e9bef" class="plotly-graph-div" style="height:900px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("c6fb06ee-6fb8-409f-a02d-dc7b0e1e9bef")) {
                    Plotly.newPlot(
                        'c6fb06ee-6fb8-409f-a02d-dc7b0e1e9bef',
                        [{"colorscale": [[0.0, "#440154"], [0.1111111111111111, "#482878"], [0.2222222222222222, "#3e4989"], [0.3333333333333333, "#31688e"], [0.4444444444444444, "#26828e"], [0.5555555555555556, "#1f9e89"], [0.6666666666666666, "#35b779"], [0.7777777777777778, "#6ece58"], [0.8888888888888888, "#b5de2b"], [1.0, "#fde725"]], "reversescale": false, "type": "heatmap", "x": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"], "y": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"], "z": [[1.0, 0.010660942645538149, -0.0016861201480870135, 0.20803373101424003, -0.010145467076855675, 0.010146427861437251, 0.02428654255096734, 0.02981995862959055, 0.5096042284323877, -0.004891877148687357, 0.49785456692658075, 0.028051167133922784, 0.29963475798369266, 0.0036335849148617645, 0.0019038955127583738, 0.053534719671228664, 0.0375097124247778, 0.6803805357911991, -0.019620818942698393, -0.021490027957098534, 0.3113087697450993, 0.21290105556556693, 0.21651336785165384, 0.20208860237515222], [0.010660942645538149, 1.0, -0.004985337352552674, -0.01680643320915765, -0.050990433654478126, 0.018354854300385467, 0.023381421528320238, 0.04613487399781132, 0.0029663348551117142, 0.03057100783714411, 0.007707058872006039, -0.032181601522581774, 0.03815343427940969, 0.022703677496334916, 0.0004732963271882408, 0.007846030957248321, 0.042142796377206686, 0.014514738706320859, 0.002452542712083261, -0.037848051005781515, -0.034054767568547466, 0.009932014960094184, -0.03322898478777197, -0.026363178228837648], [-0.0016861201480870135, -0.004985337352552674, 1.0, 0.021041825624123364, 0.032916407199243866, -0.016075326996493465, 0.031130585611766015, 0.008783279886444232, 0.00530273055449623, -0.0036688391677204214, -0.01701444474527587, 0.027472863548884053, -0.029250804197292986, 0.04023537745919304, 0.02710961848197604, 0.006557474646578711, 0.044871998853181376, 0.004628425863783853, -0.036942234339916, -0.02655600410656897, 0.009507719899011271, 0.018844999108945663, 0.010028835943115365, 0.014406048430629939], [0.20803373101424003, -0.01680643320915765, 0.021041825624123364, 1.0, 0.042070093029328424, -0.027128313256511653, 0.016774828880960865, 0.042437634318608114, 0.10158888624850132, -0.011296116736574009, 0.09496067704188856, -0.026084197160047597, 0.12631656017668416, -0.011110940860654199, -0.024538791164833422, -0.009118376696381497, 0.01842222020401694, 0.14827969653887266, -0.025100241137933486, 0.009819189309781843, 0.06911369603133874, 0.060235554120695735, 0.05425433359132035, 0.06906537825528397], [-0.010145467076855675, -0.050990433654478126, 0.032916407199243866, 0.042070093029328424, 1.0, 0.017620802485588, 0.035179212418711185, -0.006887922987035419, -0.018519193974228312, -0.046246734939416716, -0.014828515873784741, 0.012648229168459414, -0.001251032039801599, -0.012943995546151018, -0.0203588251469522, -0.06986141146763623, 0.06222669251362312, -0.014365198461300408, 0.02360316959192501, 0.010308641437966677, -0.011240463708114035, -0.008416311998425857, -0.009019064206633747, -0.009196645292770457], [0.010146427861437251, 0.018354854300385467, -0.016075326996493465, -0.027128313256511653, 0.017620802485588, 1.0, -0.04985695620300049, -0.008277598171759227, 0.001211699448913975, -0.006784352599075191, -0.006259087754780221, 0.03759962286571541, 0.012594323218943297, -0.03170119524179215, -0.029547952297681358, 0.0076653835410744765, 0.0034321577550576604, -0.0026930703885341364, -0.019359308347132335, 0.027627295460115407, 0.0014575491911166178, 0.018007460142878566, 0.016193605568453423, -0.0049987226281075076], [0.02428654255096734, 0.023381421528320238, 0.031130585611766015, 0.016774828880960865, 0.035179212418711185, -0.04985695620300049, 1.0, 0.04286064097152996, -0.027853486405547332, -0.07133462437378851, -0.015794304380892906, -0.015296749550649706, 0.022156883390196815, -0.009061986253740113, -0.0021716974278102176, 0.0013304527859505607, 0.05026339906511977, -0.002333681823322244, -0.008547685209270621, -0.004607233750264436, -0.019581616209121213, -0.02410622020878486, -0.02671558606433533, -0.020123200184066038], [0.02981995862959055, 0.04613487399781132, 0.008783279886444232, 0.042437634318608114, -0.006887922987035419, -0.008277598171759227, 0.04286064097152996, 1.0, -0.012629882671190917, -0.02147591033530702, -0.015271490778732858, -0.016322079053317184, 0.01501241324311095, -0.017204572244480576, -0.029071333439070115, 0.03429682061119727, 0.021522640378023945, -0.005533182057407529, -0.015337825759428998, -0.014616593162761336, -0.02135542697962983, 0.008716963497611708, -0.024184292365178967, 0.025975807949088905], [0.5096042284323877, 0.0029663348551117142, 0.00530273055449623, 0.10158888624850132, -0.018519193974228312, 0.001211699448913975, -0.027853486405547332, -0.012629882671190917, 1.0, -0.0019437080267456803, 0.950299913479854, 0.039562951045684296, 0.14250112381048835, -0.03473049227941859, -0.021222082108856632, 0.021641510532591626, 0.013983910528615518, 0.7822078045362802, -0.01819055019354957, 0.03781774559666627, 0.5347386873756353, 0.3894467328766716, 0.35388534696410395, 0.3752806077657263], [-0.004891877148687357, 0.03057100783714411, -0.0036688391677204214, -0.011296116736574009, -0.046246734939416716, -0.006784352599075191, -0.07133462437378851, -0.02147591033530702, -0.0019437080267456803, 1.0, -0.007156742355912731, 0.0006439169427039258, -0.05569942601274601, 0.020002039364081937, 0.0022971970637801426, -0.012453593161926874, 0.01069022612075585, -0.02018507268501687, -0.005779334958609058, -0.019458710212974534, -0.003802627948288062, -0.0023047852298301056, -0.018213567810190918, -0.02765621388428996], [0.49785456692658075, 0.007707058872006039, -0.01701444474527587, 0.09496067704188856, -0.014828515873784741, -0.006259087754780221, -0.015794304380892906, -0.015271490778732858, 0.950299913479854, -0.007156742355912731, 1.0, 0.034813626134121846, 0.14951521598969764, -0.027268586440314498, -0.017120138237390704, 0.025873436137557375, 0.005407676696812226, 0.7728932462543548, -0.021736276823893293, 0.030683081556940366, 0.5142848257331964, 0.36381766692870654, 0.34497763816542787, 0.3440788832587207], [0.028051167133922784, -0.032181601522581774, 0.027472863548884053, -0.026084197160047597, 0.012648229168459414, 0.03759962286571541, -0.015296749550649706, -0.016322079053317184, 0.039562951045684296, 0.0006439169427039258, 0.034813626134121846, 1.0, 0.017521353415571556, -0.006429345946711349, -0.00981142848936072, -0.004085329337519465, -0.034322830206661034, 0.026442471176015413, 0.0014668806322857696, 0.007963157516976974, -0.02365510670617723, -0.012814874370471655, 0.0015667995146762359, -0.036745905336769295], [0.29963475798369266, 0.03815343427940969, -0.029250804197292986, 0.12631656017668416, -0.001251032039801599, 0.012594323218943297, 0.022156883390196815, 0.01501241324311095, 0.14250112381048835, -0.05569942601274601, 0.14951521598969764, 0.017521353415571556, 1.0, -0.010238309359925531, -0.014094872753535341, 0.052733048564885456, 0.03007547509689753, 0.23763858978479374, -0.06605407172783631, -0.008365684790058657, -0.11842134024259048, -0.09075393370080816, -0.03681389238350813, -0.11031915543773214], [0.0036335849148617645, 0.022703677496334916, 0.04023537745919304, -0.011110940860654199, -0.012943995546151018, -0.03170119524179215, -0.009061986253740113, -0.017204572244480576, -0.03473049227941859, 0.020002039364081937, -0.027268586440314498, -0.006429345946711349, -0.010238309359925531, 1.0, 0.7735499964012668, -0.0404900810570771, 0.00752774782052024, -0.02060848761769149, -0.005221012351720988, -0.0032796360093679373, -0.03599126243195386, -0.0015200265442777384, -0.022154312598866084, -0.011985248472361218], [0.0019038955127583738, 0.0004732963271882408, 0.02710961848197604, -0.024538791164833422, -0.0203588251469522, -0.029547952297681358, -0.0021716974278102176, -0.029071333439070115, -0.021222082108856632, 0.0022971970637801426, -0.017120138237390704, -0.00981142848936072, -0.014094872753535341, 0.7735499964012668, 1.0, -0.03135145544245493, 0.003506471614809925, 0.006743667905952884, -0.015578881739137221, 0.002572361317682325, 0.003435126115923952, 0.03498626040719472, 0.017896066144799612, 0.022827168908479786], [0.053534719671228664, 0.007846030957248321, 0.006557474646578711, -0.009118376696381497, -0.06986141146763623, 0.0076653835410744765, 0.0013304527859505607, 0.03429682061119727, 0.021641510532591626, -0.012453593161926874, 0.025873436137557375, -0.004085329337519465, 0.052733048564885456, -0.0404900810570771, -0.03135145544245493, 1.0, -0.04595249071656096, 0.02405429182134115, 0.0024965263921170973, 0.01960440570396862, 0.019366786877455352, -0.015122914881937524, 0.033492502069353614, -0.0008674968446255921], [0.0375097124247778, 0.042142796377206686, 0.044871998853181376, 0.01842222020401694, 0.06222669251362312, 0.0034321577550576604, 0.05026339906511977, 0.021522640378023945, 0.013983910528615518, 0.01069022612075585, 0.005407676696812226, -0.034322830206661034, 0.03007547509689753, 0.00752774782052024, 0.003506471614809925, -0.04595249071656096, 1.0, 0.01013596931890164, 0.011274069611249013, 0.004128730002871822, 0.015058008028094429, 0.05081787275393108, 0.014352184864355582, 0.02469822656302997], [0.6803805357911991, 0.014514738706320859, 0.004628425863783853, 0.14827969653887266, -0.014365198461300408, -0.0026930703885341364, -0.002333681823322244, -0.005533182057407529, 0.7822078045362802, -0.02018507268501687, 0.7728932462543548, 0.026442471176015413, 0.23763858978479374, -0.02060848761769149, 0.006743667905952884, 0.02405429182134115, 0.01013596931890164, 1.0, -0.03566157127961898, 0.0010076456218964744, 0.6281331552682479, 0.4603646380118082, 0.40485775850255995, 0.4591883970831438], [-0.019620818942698393, 0.002452542712083261, -0.036942234339916, -0.025100241137933486, 0.02360316959192501, -0.019359308347132335, -0.008547685209270621, -0.015337825759428998, -0.01819055019354957, -0.005779334958609058, -0.021736276823893293, 0.0014668806322857696, -0.06605407172783631, -0.005221012351720988, -0.015578881739137221, 0.0024965263921170973, 0.011274069611249013, -0.03566157127961898, 1.0, 0.028072206603628903, 0.003568665678427497, -0.005737504337956901, -0.00206653603800598, -0.0040955260212261965], [-0.021490027957098534, -0.037848051005781515, -0.02655600410656897, 0.009819189309781843, 0.010308641437966677, 0.027627295460115407, -0.004607233750264436, -0.014616593162761336, 0.03781774559666627, -0.019458710212974534, 0.030683081556940366, 0.007963157516976974, -0.008365684790058657, -0.0032796360093679373, 0.002572361317682325, 0.01960440570396862, 0.004128730002871822, 0.0010076456218964744, 0.028072206603628903, 1.0, 0.012089185354581271, 0.04985649792220411, 0.008941249141234248, 0.002759440242340434], [0.3113087697450993, -0.034054767568547466, 0.009507719899011271, 0.06911369603133874, -0.011240463708114035, 0.0014575491911166178, -0.019581616209121213, -0.02135542697962983, 0.5347386873756353, -0.003802627948288062, 0.5142848257331964, -0.02365510670617723, -0.11842134024259048, -0.03599126243195386, 0.003435126115923952, 0.019366786877455352, 0.015058008028094429, 0.6281331552682479, 0.003568665678427497, 0.012089185354581271, 1.0, 0.7587537366134616, 0.618408865217602, 0.7692124251006991], [0.21290105556556693, 0.009932014960094184, 0.018844999108945663, 0.060235554120695735, -0.008416311998425857, 0.018007460142878566, -0.02410622020878486, 0.008716963497611708, 0.3894467328766716, -0.0023047852298301056, 0.36381766692870654, -0.012814874370471655, -0.09075393370080816, -0.0015200265442777384, 0.03498626040719472, -0.015122914881937524, 0.05081787275393108, 0.4603646380118082, -0.005737504337956901, 0.04985649792220411, 0.7587537366134616, 1.0, 0.5480562476995158, 0.7143647616385903], [0.21651336785165384, -0.03322898478777197, 0.010028835943115365, 0.05425433359132035, -0.009019064206633747, 0.016193605568453423, -0.02671558606433533, -0.024184292365178967, 0.35388534696410395, -0.018213567810190918, 0.34497763816542787, 0.0015667995146762359, -0.03681389238350813, -0.022154312598866084, 0.017896066144799612, 0.033492502069353614, 0.014352184864355582, 0.40485775850255995, -0.00206653603800598, 0.008941249141234248, 0.618408865217602, 0.5480562476995158, 1.0, 0.5102236357788096], [0.20208860237515222, -0.026363178228837648, 0.014406048430629939, 0.06906537825528397, -0.009196645292770457, -0.0049987226281075076, -0.020123200184066038, 0.025975807949088905, 0.3752806077657263, -0.02765621388428996, 0.3440788832587207, -0.036745905336769295, -0.11031915543773214, -0.011985248472361218, 0.022827168908479786, -0.0008674968446255921, 0.02469822656302997, 0.4591883970831438, -0.0040955260212261965, 0.002759440242340434, 0.7692124251006991, 0.7143647616385903, 0.5102236357788096, 1.0]]}],
                        {"height": 900, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Pearson Correlation of numerical features"}, "width": 900, "xaxis": {"nticks": 36, "ticks": ""}, "yaxis": {"ticks": ""}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('c6fb06ee-6fb8-409f-a02d-dc7b0e1e9bef');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Further Takeaway: A lot of the columns seem to be poorly correlated with one another which is good for our model. So there is no need for further feature space reduction with e.g PCA



```python
#Further exploration

numerical = [u'Age', u'DailyRate',  u'JobSatisfaction',
       u'MonthlyIncome', u'PerformanceRating',
        u'WorkLifeBalance', u'YearsAtCompany', u'Attrition_numerical']

g = sns.pairplot(df[numerical], hue='Attrition_numerical', palette='seismic', diag_kind = 'kde',diag_kws=dict(shade=True))
g.set(xticklabels=[])
```




    <seaborn.axisgrid.PairGrid at 0x1a26d1ff28>




![png](output_56_1.png)


## 2.5.2 Bi-Variate Analysis


```python

f, axes = plt.subplots(3, 3, figsize=(10, 8), 
                       sharex=False, sharey=False)
s = np.linspace(0, 3, 10)


cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
x=df['Age']
y=df['TotalWorkingYears']

ax1=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[0,0], cut=5)
ax1.set_title('Age vs. Working Years')
ax1.set_xlabel('')
ax1.set_ylabel('')


cmap = sns.cubehelix_palette(start=0.33, light=1, as_cmap=True)
x=df['Age']
y=df['DailyRate']
ax2=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[0,1])
ax2.set_title('Age vs. Daily Rate')
ax2.set_xlabel('')
ax2.set_ylabel('')


cmap = sns.cubehelix_palette(start=0.66, light=1, as_cmap=True)
x=df['YearsInCurrentRole']
y=df['Age']
ax3=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[0,2])
ax3.set_title('YearsInRole vs. Age')
ax3.set_xlabel('')
ax3.set_ylabel('')


cmap = sns.cubehelix_palette(start=0.99, light=1, as_cmap=True)
x=df['DailyRate']
y=df['DistanceFromHome']
ax4=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[1,0])
ax4.set_title('DailyRate vs. Distance from Home')
ax4.set_xlabel('')
ax4.set_ylabel('')


cmap = sns.cubehelix_palette(start=1.33, light=1, as_cmap=True)
x=df['DailyRate']
y=df['JobSatisfaction']
ax5=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[1,1])
ax5.set_title('Daily Rate vs. Job Satisfaction')
ax5.set_xlabel('')
ax5.set_ylabel('')


cmap = sns.cubehelix_palette(start=1.66, light=1, as_cmap=True)
x=df['YearsAtCompany']
y=df['JobSatisfaction']
ax6=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[1,2])
ax6.set_title('Years at Company Vs. Job Satisfaction')
ax6.set_xlabel('')
ax6.set_ylabel('')


cmap = sns.cubehelix_palette(start=1.9999, light=1, as_cmap=True)
x=df['YearsAtCompany']
y=df['DailyRate']
ax7=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[2,0])
ax7.set_title('Company Years vs. Daily Rate')
ax7.set_xlabel('')
ax7.set_ylabel('')


cmap = sns.cubehelix_palette(start=2.33, light=1, as_cmap=True)
x=df['RelationshipSatisfaction']
y=df['YearsWithCurrManager']
ax8=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[2,1])
ax8.set_title('Satisfaction vs. Manager Years')
ax8.set_xlabel('')
ax8.set_ylabel('')


cmap = sns.cubehelix_palette(start=2.666, light=1, as_cmap=True)
x=df['WorkLifeBalance']
y=df['RelationshipSatisfaction']
ax9=sns.kdeplot(x,y, shade=True, cmap=cmap, ax=axes[2,2])
ax9.set_title('WorkLife Balance Vs. Satisfaction')
ax9.set_xlabel('')
ax9.set_ylabel('')

f.tight_layout()
```


![png](output_58_0.png)



```python
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)

sns.set(style="dark")
fig=plt.figure(figsize=(12,10))
fig.tight_layout(pad=1.0)

ax1 = plt.subplot(gs[0, :2])
ax2 = plt.subplot(gs[0, 2:4])
ax3 = plt.subplot(gs[1, 0:2])
ax4 = plt.subplot(gs[1, 2:4])


f = sns.scatterplot(x=df['TotalWorkingYears'], y=df['MonthlyIncome'], ax=ax1, s=25, color='#BF0F9F', alpha=0.5)
f.set_xlabel('Working Years')
f.set_ylabel('Monthly Income')
f.set_title('Positive Correlation',fontweight="bold", size=14)


g = sns.stripplot(x=df['PerformanceRating'], y=df['PercentSalaryHike'], ax=ax2, s=10, jitter=1, alpha=0.5, color='#BF0F9F')
g.set_xlabel('Performance Rating')
g.set_ylabel('Percent Salary Hike')
g.title.set_text('Positive Correlation')
g.set_title('Positive Correlation',fontweight="bold", size=14)

h = sns.scatterplot(x=df['Age'], y=df['MonthlyIncome'], ax=ax4, s=25, color='#BF0F9F', alpha=0.5)
h.set_xlabel('Working Years')
h.set_ylabel('Monthly Income')


i = sns.boxplot(x=df['YearsWithCurrManager'], y= df['YearsSinceLastPromotion'], ax=ax3)
i = sns.scatterplot(x=df['YearsWithCurrManager'], y= df['YearsSinceLastPromotion'], alpha=0.5, ax=ax3)
i.set_xlabel('Years With Current Manager')
i.set_ylabel('Years Since Last Promotion')
```




    Text(0,0.5,'Years Since Last Promotion')




![png](output_59_1.png)


# 3. Feature Engineering 

## 3.1 Encoding the categorical values


```python
#Get all categorical features and saving them in a list
categorical=[]

for a in df.columns:
    if df[a].dtypes=='object':
        categorical.append(a)
```


```python
df.drop('Attrition_numerical', axis=1, inplace=True)
```


```python
numerical = df.columns.difference(categorical)
```


```python
df[categorical].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition</th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>EducationField</th>
      <th>Gender</th>
      <th>JobRole</th>
      <th>MaritalStatus</th>
      <th>Over18</th>
      <th>OverTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Sales Executive</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>Research Scientist</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>Other</td>
      <td>Male</td>
      <td>Laboratory Technician</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>No</td>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Research Scientist</td>
      <td>Married</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>No</td>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>Medical</td>
      <td>Male</td>
      <td>Laboratory Technician</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_cat = df[categorical]
df_cat.drop('Attrition', axis=1, inplace=True)
```


```python
df_cat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BusinessTravel</th>
      <th>Department</th>
      <th>EducationField</th>
      <th>Gender</th>
      <th>JobRole</th>
      <th>MaritalStatus</th>
      <th>Over18</th>
      <th>OverTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Travel_Rarely</td>
      <td>Sales</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Sales Executive</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>Life Sciences</td>
      <td>Male</td>
      <td>Research Scientist</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>Other</td>
      <td>Male</td>
      <td>Laboratory Technician</td>
      <td>Single</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Travel_Frequently</td>
      <td>Research &amp; Development</td>
      <td>Life Sciences</td>
      <td>Female</td>
      <td>Research Scientist</td>
      <td>Married</td>
      <td>Y</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Travel_Rarely</td>
      <td>Research &amp; Development</td>
      <td>Medical</td>
      <td>Male</td>
      <td>Laboratory Technician</td>
      <td>Married</td>
      <td>Y</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Numerical encoding of the data via pd.get_dummies
df_cat = pd.get_dummies(df_cat)
```


```python
df_num = df[numerical]
```


```python
#Creating the final dataset with numerical data only
df_final = pd.concat([df_num, df_cat], axis=1)
```


```python
df_final.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>DailyRate</th>
      <th>DistanceFromHome</th>
      <th>Education</th>
      <th>EmployeeNumber</th>
      <th>EnvironmentSatisfaction</th>
      <th>HourlyRate</th>
      <th>JobInvolvement</th>
      <th>JobLevel</th>
      <th>JobSatisfaction</th>
      <th>...</th>
      <th>JobRole_Research Director</th>
      <th>JobRole_Research Scientist</th>
      <th>JobRole_Sales Executive</th>
      <th>JobRole_Sales Representative</th>
      <th>MaritalStatus_Divorced</th>
      <th>MaritalStatus_Married</th>
      <th>MaritalStatus_Single</th>
      <th>Over18_Y</th>
      <th>OverTime_No</th>
      <th>OverTime_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41</td>
      <td>1102</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>94</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49</td>
      <td>279</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>61</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1373</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>92</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 53 columns</p>
</div>




```python
target_map = {'Yes':1, 'No':0}
target = df["Attrition"].apply(lambda x: target_map[x])
target.head(3)
```




    0    1
    1    0
    2    1
    Name: Attrition, dtype: int64




```python
#Checking if the data of the target variable is distributed equally. Imbalances in the target variable could have a negative effect when it comes to building the model.
data=[go.Bar(x=df['Attrition'].value_counts().index, y=df['Attrition'].value_counts().values)]
py.iplot(data)


```


<div>
        
        
            <div id="79182931-9c70-4980-9f70-fc7aafc7c46f" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("79182931-9c70-4980-9f70-fc7aafc7c46f")) {
                    Plotly.newPlot(
                        '79182931-9c70-4980-9f70-fc7aafc7c46f',
                        [{"type": "bar", "x": ["No", "Yes"], "y": [1233, 237]}],
                        {"template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('79182931-9c70-4980-9f70-fc7aafc7c46f');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
df["Attrition"].value_counts().index
```




    Index(['No', 'Yes'], dtype='object')



## 3.2 Work with Skewness

Our target variable is skewed and not distributed equally in its categories. Therefore, we'll use the Smote Function as an upsampling method to avoid a loss of information.


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

train, test, target_train, target_val = train_test_split(df_final, 
                                                         target, 
                                                         train_size= 0.80,
                                                         random_state=0);
```


```python
#SMOTE to oversample due to the skewness in target. I used a upsampling method to avoid a loss of information.

sampler=SMOTE(random_state=0)
smote_train, smote_target = sampler.fit_sample(train,target_train)
```

# 4. Analysis & Implementation of Machine Learning Models

In the following part we'll implement to different Machine Learning Models. First we'll use a Random Forest Classifier. After that we'll try the Gradient Boosting Classifier. Both models will be compared with eachother and include further visualizations.

## 4.1 Random Forest Classifier


```python
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : 0,
    'verbose': 0
}
```


```python
rf = RandomForestClassifier(**rf_params)
```


```python
fit = rf.fit(smote_train, smote_target)
```


```python
rf_predictions = rf.predict(test)
```


```python
#Checking the Accuracy of the model

print("Accuracy score: {}".format(accuracy_score(target_val, rf_predictions)))
print("="*80)
print(classification_report(target_val, rf_predictions))
```

    Accuracy score: 0.8537414965986394
    ================================================================================
                  precision    recall  f1-score   support
    
               0       0.89      0.95      0.92       245
               1       0.59      0.39      0.47        49
    
        accuracy                           0.85       294
       macro avg       0.74      0.67      0.69       294
    weighted avg       0.84      0.85      0.84       294
    



```python
#Using Randomized Search for Hyperparameter Tuning. Using Randomized Search instead of GridSearch to keep the runtime low

from sklearn.model_selection import RandomizedSearchCV
```


```python
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [1,2,3,4,5,6,7,8,9,10]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
```


```python
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
```


```python
rf = RandomForestClassifier()
```


```python
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
```


```python
rf_random.fit(smote_train, smote_target)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  33 tasks      | elapsed:   30.5s
    [Parallel(n_jobs=-1)]: Done 154 tasks      | elapsed:  2.6min
    [Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed:  4.6min finished





    RandomizedSearchCV(cv=3, error_score=nan,
                       estimator=RandomForestClassifier(bootstrap=True,
                                                        ccp_alpha=0.0,
                                                        class_weight=None,
                                                        criterion='gini',
                                                        max_depth=None,
                                                        max_features='auto',
                                                        max_leaf_nodes=None,
                                                        max_samples=None,
                                                        min_impurity_decrease=0.0,
                                                        min_impurity_split=None,
                                                        min_samples_leaf=1,
                                                        min_samples_split=2,
                                                        min_weight_fraction_leaf=0.0,
                                                        n_estimators=100,
                                                        n_jobs...
                       iid='deprecated', n_iter=100, n_jobs=-1,
                       param_distributions={'bootstrap': [True, False],
                                            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                          10],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 4],
                                            'min_samples_split': [2, 5, 10],
                                            'n_estimators': [100, 311, 522, 733,
                                                             944, 1155, 1366, 1577,
                                                             1788, 2000]},
                       pre_dispatch='2*n_jobs', random_state=42, refit=True,
                       return_train_score=False, scoring=None, verbose=2)




```python
rf_random.best_params_
```




    {'n_estimators': 522,
     'min_samples_split': 2,
     'min_samples_leaf': 2,
     'max_features': 'sqrt',
     'max_depth': 10,
     'bootstrap': True}




```python
pred = rf_random.predict(test)
```


```python
#Checking the Accuracy of the model

print("Accuracy score: {}".format(accuracy_score(target_val, pred)))
print("="*80)
print(classification_report(target_val, pred))
```

    Accuracy score: 0.8775510204081632
    ================================================================================
                  precision    recall  f1-score   support
    
               0       0.88      0.99      0.93       245
               1       0.84      0.33      0.47        49
    
        accuracy                           0.88       294
       macro avg       0.86      0.66      0.70       294
    weighted avg       0.87      0.88      0.85       294
    


**Takeaway**

The Random Forest Classifier returns an accuracy of 87% for its predictions. We have to keep in mind that the distribution of the target variable used to be skewed with 84% of yes and 16% of no's. So the model is slightly better than random guessing

It would be more informative to balance out the precision and recall scores as show in the classification report outputs. Where it falls down to the business considerations over whether one should prioritise for a metric over the other - i.e. Precision vs Recall.


```python
#Having a look on the importances of the features + visualization

fit.feature_importances_
```




    array([1.57241423e-02, 3.48879898e-03, 5.56675345e-03, 6.16420881e-03,
           2.18830155e-03, 1.46836861e-02, 2.06315771e-03, 1.25276173e-02,
           4.16983979e-02, 3.65854687e-02, 4.02121093e-02, 3.12269254e-03,
           7.61602157e-03, 1.65288620e-03, 8.64371079e-04, 9.51406237e-03,
           6.35993438e-02, 2.39162966e-02, 3.28221791e-03, 1.51724826e-02,
           1.65555001e-02, 1.59447097e-02, 2.47883001e-03, 1.64212820e-02,
           3.68696987e-03, 5.32048717e-02, 1.95237532e-02, 1.13733139e-03,
           3.27350167e-02, 2.38222883e-02, 5.12744731e-04, 1.93492942e-02,
           5.23466422e-03, 3.01819199e-02, 3.74707748e-04, 2.94265540e-03,
           7.51530510e-03, 6.66618463e-03, 3.70653183e-03, 1.49373073e-03,
           1.60014353e-02, 1.72285660e-04, 3.78631704e-03, 1.22798609e-04,
           5.49912087e-03, 1.09134342e-02, 8.94683374e-03, 1.01789410e-02,
           1.58734097e-02, 7.40118680e-02, 0.00000000e+00, 1.42583377e-01,
           1.38778871e-01])




```python
trace = go.Scatter(y=fit.feature_importances_, x=df_final.columns.values, 
                   mode='markers',
                   marker=dict(sizemode = 'diameter',
                               sizeref = 1,
                               size = 13,
                               color = fit.feature_importances_,
                               colorscale='Portland',
                               showscale=True),
                   text = df_final.columns.values), 

data=trace

layout = go.Layout(autosize=True, title="Feature Importance", hovermode='closest',
                   yaxis_title="Feature Importance", height=800, width=1100, showlegend=False,
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   xaxis= dict(ticklen= 5,
                               showgrid=False,
                               zeroline=False,),
                   yaxis = dict(ticklen= 5,
                               showgrid=False,
                               zeroline=False,))



fig = go.Figure(data=data, layout=layout)
fig.show()

```


<div>
        
        
            <div id="f2568d29-ed2c-4103-b1e6-fa645e181092" class="plotly-graph-div" style="height:800px; width:1100px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("f2568d29-ed2c-4103-b1e6-fa645e181092")) {
                    Plotly.newPlot(
                        'f2568d29-ed2c-4103-b1e6-fa645e181092',
                        [{"marker": {"color": [0.015724142265165362, 0.003488798976796352, 0.0055667534491052025, 0.0061642088097037615, 0.002188301548443185, 0.01468368614426226, 0.0020631577090428495, 0.012527617274164682, 0.0416983979151971, 0.036585468687350686, 0.04021210929111153, 0.003122692539214479, 0.007616021570236651, 0.0016528861953108607, 0.0008643710786356187, 0.009514062372306824, 0.06359934383643528, 0.023916296571144424, 0.003282217910270404, 0.015172482639846725, 0.016555500083978016, 0.015944709723199196, 0.002478830006396935, 0.01642128202577514, 0.0036869698690777867, 0.05320487165430628, 0.01952375315420688, 0.0011373313861685775, 0.03273501672891052, 0.023822288279582233, 0.0005127447309905491, 0.019349294219832294, 0.005234664219392625, 0.030181919861008596, 0.00037470774750337626, 0.002942655402658699, 0.0075153051044757, 0.006666184634589566, 0.0037065318317320234, 0.0014937307275851546, 0.01600143532562941, 0.00017228565959466938, 0.003786317042402305, 0.00012279860884405148, 0.005499120868278503, 0.010913434216741728, 0.008946833740626431, 0.01017894096885857, 0.015873409662225002, 0.07401186803085191, 0.0, 0.14258337653262343, 0.1387788711682095], "colorscale": [[0.0, "rgb(12,51,131)"], [0.25, "rgb(10,136,186)"], [0.5, "rgb(242,211,56)"], [0.75, "rgb(242,143,56)"], [1.0, "rgb(217,30,30)"]], "showscale": true, "size": 13, "sizemode": "diameter", "sizeref": 1}, "mode": "markers", "text": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely", "Department_Human Resources", "Department_Research & Development", "Department_Sales", "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing", "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree", "Gender_Female", "Gender_Male", "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative", "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", "Over18_Y", "OverTime_No", "OverTime_Yes"], "type": "scatter", "x": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely", "Department_Human Resources", "Department_Research & Development", "Department_Sales", "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing", "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree", "Gender_Female", "Gender_Male", "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative", "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", "Over18_Y", "OverTime_No", "OverTime_Yes"], "y": [0.015724142265165362, 0.003488798976796352, 0.0055667534491052025, 0.0061642088097037615, 0.002188301548443185, 0.01468368614426226, 0.0020631577090428495, 0.012527617274164682, 0.0416983979151971, 0.036585468687350686, 0.04021210929111153, 0.003122692539214479, 0.007616021570236651, 0.0016528861953108607, 0.0008643710786356187, 0.009514062372306824, 0.06359934383643528, 0.023916296571144424, 0.003282217910270404, 0.015172482639846725, 0.016555500083978016, 0.015944709723199196, 0.002478830006396935, 0.01642128202577514, 0.0036869698690777867, 0.05320487165430628, 0.01952375315420688, 0.0011373313861685775, 0.03273501672891052, 0.023822288279582233, 0.0005127447309905491, 0.019349294219832294, 0.005234664219392625, 0.030181919861008596, 0.00037470774750337626, 0.002942655402658699, 0.0075153051044757, 0.006666184634589566, 0.0037065318317320234, 0.0014937307275851546, 0.01600143532562941, 0.00017228565959466938, 0.003786317042402305, 0.00012279860884405148, 0.005499120868278503, 0.010913434216741728, 0.008946833740626431, 0.01017894096885857, 0.015873409662225002, 0.07401186803085191, 0.0, 0.14258337653262343, 0.1387788711682095]}],
                        {"autosize": true, "height": 800, "hovermode": "closest", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Feature Importance"}, "width": 1100, "xaxis": {"showgrid": false, "ticklen": 5, "zeroline": false}, "yaxis": {"showgrid": false, "ticklen": 5, "title": {"text": "Feature Importance"}, "zeroline": false}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('f2568d29-ed2c-4103-b1e6-fa645e181092');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



```python
#Visualization of Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 


dtree = DecisionTreeClassifier(max_depth=4)
dtree.fit(train, target_train)
y_pred = dtree.predict(test)
```


```python
features = list(df_final.columns.values)

dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())

```




![png](output_100_0.png)



## 4.2 Gradient Boosting Classifier

As an alternative to the Random Forest I tried the Gradient Boosted Classifier as another ensemble technique


```python
gb_params ={
    'n_estimators': 1500,
    'max_features': 0.9,
    'learning_rate' : 0.25,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'subsample': 1,
    'max_features' : 'sqrt',
    'verbose': 0
}
```


```python
gb = GradientBoostingClassifier(**gb_params)
gb.fit(smote_train, smote_target)
gb_predictions = gb.predict(test)
```


```python
print(accuracy_score(target_val, gb_predictions))
print(classification_report(target_val, gb_predictions))
```

    0.8571428571428571
                  precision    recall  f1-score   support
    
               0       0.87      0.97      0.92       245
               1       0.65      0.31      0.42        49
    
        accuracy                           0.86       294
       macro avg       0.76      0.64      0.67       294
    weighted avg       0.84      0.86      0.83       294
    



```python
#Hyperparameter Tuning with Randomized Search

params ={
    'n_estimators': [500, 700, 1000, 1100, 1200,1300,1400,1500],
    'max_features': [1,2,3,4,5,6,7,8,9],
    'learning_rate' : [0.001,0.01,0.1,0.2,0.25,0.3],
    'max_depth': [2,3,4,4.5,5,6,7,8,9,10],
    'min_samples_leaf': [1,2,3,4,5],
    'max_features' : ['auto','sqrt']
}
```


```python
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = params, n_iter = 100, cv = 3, verbose=0, random_state=1, n_jobs = -1)
```


```python
gb_random.fit(smote_train, smote_target)
```




    RandomizedSearchCV(cv=3, error_score=nan,
                       estimator=GradientBoostingClassifier(ccp_alpha=0.0,
                                                            criterion='friedman_mse',
                                                            init=None,
                                                            learning_rate=0.25,
                                                            loss='deviance',
                                                            max_depth=4,
                                                            max_features='sqrt',
                                                            max_leaf_nodes=None,
                                                            min_impurity_decrease=0.0,
                                                            min_impurity_split=None,
                                                            min_samples_leaf=2,
                                                            min_samples_split=2,
                                                            min_weight_fraction_leaf=0.0,
                                                            n_estimators=1500,
                                                            n...
                       iid='deprecated', n_iter=100, n_jobs=-1,
                       param_distributions={'learning_rate': [0.001, 0.01, 0.1, 0.2,
                                                              0.25, 0.3],
                                            'max_depth': [2, 3, 4, 4.5, 5, 6, 7, 8,
                                                          9, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 3, 4, 5],
                                            'n_estimators': [500, 700, 1000, 1100,
                                                             1200, 1300, 1400,
                                                             1500]},
                       pre_dispatch='2*n_jobs', random_state=1, refit=True,
                       return_train_score=False, scoring=None, verbose=0)




```python
gb_random.best_params_
```




    {'n_estimators': 700,
     'min_samples_leaf': 1,
     'max_features': 'sqrt',
     'max_depth': 10,
     'learning_rate': 0.01}




```python
pred = gb_random.predict(test)
```


```python
print(accuracy_score(target_val, pred))
print(classification_report(target_val, pred))
```

    0.8775510204081632
                  precision    recall  f1-score   support
    
               0       0.88      0.99      0.93       245
               1       0.88      0.31      0.45        49
    
        accuracy                           0.88       294
       macro avg       0.88      0.65      0.69       294
    weighted avg       0.88      0.88      0.85       294
    



```python
params ={
    'n_estimators': [1200,1300,1400],
    'max_features': [6,7,8,9],
    'learning_rate' : [0.01],
    'max_depth': [2,3,4,4.5,5,6,7,8,9,10],
    'min_samples_leaf': [1,2,3,4,5],
    'max_features' : ['auto','sqrt']
}
```


```python
gb_random = RandomizedSearchCV(estimator = gb, param_distributions = params, n_iter = 100, cv = 3, verbose=0, random_state=1, n_jobs = -1)
```


```python
gb_random.fit(smote_train, smote_target)
```




    RandomizedSearchCV(cv=3, error_score=nan,
                       estimator=GradientBoostingClassifier(ccp_alpha=0.0,
                                                            criterion='friedman_mse',
                                                            init=None,
                                                            learning_rate=0.25,
                                                            loss='deviance',
                                                            max_depth=4,
                                                            max_features='sqrt',
                                                            max_leaf_nodes=None,
                                                            min_impurity_decrease=0.0,
                                                            min_impurity_split=None,
                                                            min_samples_leaf=2,
                                                            min_samples_split=2,
                                                            min_weight_fraction_leaf=0.0,
                                                            n_estimators=1500,
                                                            n...
                                                            validation_fraction=0.1,
                                                            verbose=0,
                                                            warm_start=False),
                       iid='deprecated', n_iter=100, n_jobs=-1,
                       param_distributions={'learning_rate': [0.01],
                                            'max_depth': [2, 3, 4, 4.5, 5, 6, 7, 8,
                                                          9, 10],
                                            'max_features': ['auto', 'sqrt'],
                                            'min_samples_leaf': [1, 2, 3, 4, 5],
                                            'n_estimators': [1200, 1300, 1400]},
                       pre_dispatch='2*n_jobs', random_state=1, refit=True,
                       return_train_score=False, scoring=None, verbose=0)




```python
gb_random.best_params_
```


```python
pred = gb_random.predict(test)
```


```python
print(accuracy_score(target_val, pred))
print(classification_report(target_val, pred))
```

**Takeaway:**

GBC returns an accuracy of 87% after tuning of hyperparameters.


```python
#Having a look on the importances of the features + visualization

gb.feature_importances_
```




    array([1.65173379e-02, 1.71652465e-02, 1.29592219e-02, 5.53293893e-03,
           1.01340656e-02, 2.06557769e-02, 6.15255147e-03, 4.33083128e-02,
           2.63758516e-02, 4.44982081e-02, 6.35431677e-02, 1.31743615e-02,
           2.14472362e-02, 6.47106669e-03, 3.77489729e-04, 1.30139964e-02,
           6.27701025e-02, 1.23863499e-02, 3.26673567e-03, 1.25958205e-02,
           3.22375622e-02, 8.44264696e-03, 6.90150290e-03, 1.74076716e-02,
           8.33376731e-03, 2.28464756e-02, 1.82223695e-02, 6.50939735e-04,
           4.04797595e-02, 3.77496341e-02, 1.16353897e-03, 3.73200881e-02,
           2.48903688e-03, 3.25503779e-02, 1.71205661e-03, 1.15351503e-03,
           1.56801412e-02, 2.04797931e-03, 5.69069697e-03, 1.35043095e-03,
           1.31826860e-02, 2.66408370e-05, 2.45505186e-03, 9.36959489e-05,
           7.84668721e-03, 1.28861366e-02, 4.08114941e-03, 1.01500547e-02,
           4.07142191e-02, 3.34009619e-02, 0.00000000e+00, 3.85602890e-02,
           1.29826398e-01])




```python
trace = go.Scatter(y=gb.feature_importances_, x=df_final.columns.values, 
                   mode='markers',
                   marker=dict(sizemode = 'diameter',
                               sizeref = 1,
                               size = 13,
                               color = gb.feature_importances_,
                               colorscale='Portland',
                               showscale=True),
                   text = df_final.columns.values), 

data=trace

layout = go.Layout(autosize=True, title="Feature Importance", hovermode='closest',
                   yaxis_title="Feature Importance", height=800, width=1100, showlegend=False,
                   paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(0,0,0,0)',
                   xaxis= dict(ticklen= 5,
                               showgrid=False,
                               zeroline=False,),
                   yaxis = dict(ticklen= 5,
                               showgrid=False,
                               zeroline=False,))



fig = go.Figure(data=data, layout=layout)
fig.show()
```


<div>
        
        
            <div id="6b4b281b-bec7-4acf-8060-23bf94735f8e" class="plotly-graph-div" style="height:800px; width:1100px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    
                if (document.getElementById("6b4b281b-bec7-4acf-8060-23bf94735f8e")) {
                    Plotly.newPlot(
                        '6b4b281b-bec7-4acf-8060-23bf94735f8e',
                        [{"marker": {"color": [0.016517337933743964, 0.017165246501858765, 0.012959221939953492, 0.005532938930875404, 0.010134065604297327, 0.020655776853888915, 0.006152551473738325, 0.043308312817920976, 0.02637585162148297, 0.04449820807381423, 0.06354316773444817, 0.013174361520140001, 0.02144723618180411, 0.006471066686769146, 0.0003774897290750603, 0.01301399637663602, 0.06277010253276381, 0.012386349881712038, 0.003266735666816216, 0.01259582054579907, 0.032237562186689994, 0.008442646957765771, 0.006901502903960975, 0.01740767162590989, 0.008333767305916338, 0.022846475620903195, 0.01822236946198609, 0.00065093973530428, 0.04047975947417087, 0.03774963414028631, 0.001163538968162161, 0.0373200880542399, 0.002489036883388458, 0.03255037786438033, 0.0017120566114754933, 0.0011535150337328368, 0.015680141166703868, 0.002047979310572082, 0.0056906969659693595, 0.0013504309536783155, 0.013182685998212337, 2.6640836988064786e-05, 0.002455051857643237, 9.369594885287964e-05, 0.007846687210320938, 0.012886136603569402, 0.0040811494084650995, 0.010150054673904677, 0.04071421910409884, 0.033400961949959514, 0.0, 0.03856028896171349, 0.12982639761353687], "colorscale": [[0.0, "rgb(12,51,131)"], [0.25, "rgb(10,136,186)"], [0.5, "rgb(242,211,56)"], [0.75, "rgb(242,143,56)"], [1.0, "rgb(217,30,30)"]], "showscale": true, "size": 13, "sizemode": "diameter", "sizeref": 1}, "mode": "markers", "text": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely", "Department_Human Resources", "Department_Research & Development", "Department_Sales", "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing", "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree", "Gender_Female", "Gender_Male", "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative", "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", "Over18_Y", "OverTime_No", "OverTime_Yes"], "type": "scatter", "x": ["Age", "DailyRate", "DistanceFromHome", "Education", "EmployeeNumber", "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "BusinessTravel_Non-Travel", "BusinessTravel_Travel_Frequently", "BusinessTravel_Travel_Rarely", "Department_Human Resources", "Department_Research & Development", "Department_Sales", "EducationField_Human Resources", "EducationField_Life Sciences", "EducationField_Marketing", "EducationField_Medical", "EducationField_Other", "EducationField_Technical Degree", "Gender_Female", "Gender_Male", "JobRole_Healthcare Representative", "JobRole_Human Resources", "JobRole_Laboratory Technician", "JobRole_Manager", "JobRole_Manufacturing Director", "JobRole_Research Director", "JobRole_Research Scientist", "JobRole_Sales Executive", "JobRole_Sales Representative", "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single", "Over18_Y", "OverTime_No", "OverTime_Yes"], "y": [0.016517337933743964, 0.017165246501858765, 0.012959221939953492, 0.005532938930875404, 0.010134065604297327, 0.020655776853888915, 0.006152551473738325, 0.043308312817920976, 0.02637585162148297, 0.04449820807381423, 0.06354316773444817, 0.013174361520140001, 0.02144723618180411, 0.006471066686769146, 0.0003774897290750603, 0.01301399637663602, 0.06277010253276381, 0.012386349881712038, 0.003266735666816216, 0.01259582054579907, 0.032237562186689994, 0.008442646957765771, 0.006901502903960975, 0.01740767162590989, 0.008333767305916338, 0.022846475620903195, 0.01822236946198609, 0.00065093973530428, 0.04047975947417087, 0.03774963414028631, 0.001163538968162161, 0.0373200880542399, 0.002489036883388458, 0.03255037786438033, 0.0017120566114754933, 0.0011535150337328368, 0.015680141166703868, 0.002047979310572082, 0.0056906969659693595, 0.0013504309536783155, 0.013182685998212337, 2.6640836988064786e-05, 0.002455051857643237, 9.369594885287964e-05, 0.007846687210320938, 0.012886136603569402, 0.0040811494084650995, 0.010150054673904677, 0.04071421910409884, 0.033400961949959514, 0.0, 0.03856028896171349, 0.12982639761353687]}],
                        {"autosize": true, "height": 800, "hovermode": "closest", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "showlegend": false, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Feature Importance"}, "width": 1100, "xaxis": {"showgrid": false, "ticklen": 5, "zeroline": false}, "yaxis": {"showgrid": false, "ticklen": 5, "title": {"text": "Feature Importance"}, "zeroline": false}},
                        {"responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('6b4b281b-bec7-4acf-8060-23bf94735f8e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# 5. Interpretation

We started this project with a detailed EDA to dig deeper into our data. This allowed us to get a first idea which circumstances can possibly lead to attrition. After engineering the features (e.g handling imbalanced target variable) we implented two models in the form of a Random Forest and a Gradient Boosting classifier and tuned its hyperparameters via Random Search. The notebook returns a 88% accuracy in its predictions. 
Important features seem to be:

- **Overtime**: As expected, overtime has a huge impact if an employee will leave the organization or not. 

- **Monthly Income**:  Income is a factor as why employees leave the organization in search for a better salary.

- **Stock Option Level**: In both models the option on stocks influences the level of attrition.


Nevertheless, there is still room for improvement e.g. more features could be engineered from the data. 
Furthermore, one could improve the outcome by for example using some form of blending or stacking of models, where a handful of classifiers votes on the outcome of the predictions and we eventually take the majority vote.



```python

```