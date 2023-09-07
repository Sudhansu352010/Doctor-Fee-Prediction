# Doctor-Fee-Prediction
![image](https://github.com/Sudhansu352010/Doctor-Fee-Prediction/assets/131376814/e873a621-a51d-4e47-af92-9db45ec52b1c)

# Objective
Create a machine learning model to predict doctor’s fee
Creating a webpage for user to get accurate prediction of fee based on multiple factors

## Web Scraping
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from selenium.webdriver.common.keys import Keys 
import matplotlib.pyplot as plt
import seaborn as sns
#Creating empty data drame
df= pd.DataFrame({'Name':[''], 'Speciality':[''], 'Degree':[''], 'Year_of_experience':[''] , 'Location':[''] , 'City':[''] ,'dp_score':[''], 'npv':[''], 'consultation_fee':['']})
#Creating list of cities from where data is to be extracted
lis=['Bangalore','Delhi','Mumbai']
#speciality of doctors whose data to be extracted
Speciality = 'chiropractor'
#for loop to take city name form lis and each time change url according to city
for i in lis:
    driver = webdriver.Chrome()
    url = f'https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22chiropractor%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city={i}'
    driver.get(url)
    time.sleep(2)
    
    scroll_pause_time = 2 # You can set your own pause time. dont slow too slow that might not able to load more data
    screen_height = driver.execute_script("return window.screen.height;")  # get the screen height of the web
    A = 1

    while True:
        # scroll one screen height each time
        driver.execute_script("window.scrollTo(0, {screen_height}*{A});".format(screen_height=screen_height, A=A))
        A += 1
        time.sleep(scroll_pause_time)
        # update scroll height each time after scrolled, as the scroll height can change after we scrolled the page
        scroll_height = driver.execute_script("return document.body.scrollHeight;")
        # Break the loop when the height we need to scroll to is larger than the total scroll height
        if (screen_height) * A > scroll_height:
            break
    #after scrolling getting the HTML of webpage and extrancting data
    soup = BeautifulSoup(driver.page_source,'lxml')
    postings = soup.find_all('div' , class_= 'u-border-general--bottom')
    for post in postings:
        try:
            link = post.find('div' , class_ = 'listing-doctor-card').find('a').get('href')
            link_full = 'https://www.practo.com'+link
            driver.get(link_full)
            soup2 = BeautifulSoup(driver.page_source,'lxml')

            #extracting name
            try:
                name = soup2.find('h1' , class_ = 'c-profile__title u-bold u-d-inlineblock').text
            except:
                pass       
            #extracting degree
            try:
                Degree = soup2.find('p' , class_ = 'c-profile__details').text
            except:
                pass
            #extracting years of experience
            try:
                Year_of_experience = soup2.find('div' , class_ = 'c-profile__details').find_all('h2')[-1].text
            except:
                pass
            #extracting location
            try:
                Location = soup2.find('h4' , class_ = 'c-profile--clinic__location').text
            except:
                pass
            #extracting dp score
            try:
                dp_score = soup2.find('span' , class_ = 'u-green-text u-bold u-large-font').text.strip()
            except:
                pass
            #extracting npv
            try:
                npv = soup2.find('span' , class_ = 'u-smallest-font u-grey_3-text').text
            except:
                pass
            #extracting consulting fee
            try:
                consultant_fee = soup2.find('span' , class_ = 'u-strike').text.strip()
            except:
                consultant_fee = soup2.find('div' , class_ = 'u-f-right u-large-font u-bold u-valign--middle u-lheight-normal').text.strip()
            #appending all data into dataframe
            df = df.append({'Name':name, 'Speciality':Speciality, 'Degree':Degree, 'Year_of_experience':Year_of_experience , 'Location':Location , 'City':i ,'dp_score':dp_score, 'npv':npv, 'consultation_fee':consultant_fee} , ignore_index = True)
        except:
            pass
df

df.nunique()

df['Speciality'].value_counts()

df.to_csv('chiropractor.csv',index=False)

### Similarly we can extract data for other speciality by doing some changes in this code:

### In line 3, assign the name of speciality to 'Speciality' variable.

### In line 6, assign the url of that speciality to 'url' variable.

### Specialities: Cardiologist, Chiropractor, Dentist, Dermatologist, Dietitian, Gastroenterologist, bariatric surgeon, Gynecologist, Infertility Specialist, Neurologist, Neurosurgeon, Ophthalmologist, Orthopedist, Pediatrician, Physiotherapist, Psychiatrist, Pulmonologist, Rheumatologists, Urologist

specialists = [
    {"Specialist": "Cardiologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22cardiologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Chiropractor", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22chiropractor%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Dentist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22dentist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Dermatologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22dermatologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Dietitian", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22dietitian%2Fnutritionist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Gastroenterologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22gastroenterologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Bariatric Surgeon", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22bariatric%20surgeon%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Gynecologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22gynecologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Infertility Specialist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22infertility%20specialist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Neurologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22neurologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Neurosurgeon", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22neurosurgeon%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Ophthalmologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22ophthalmologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Orthopedist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22orthopedist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Pediatrician", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22pediatrician%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Physiotherapist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22physiotherapist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Psychiatrist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22psychiatrist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Pulmonologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22pulmonologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Rheumatologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22rheumatologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"},
    {"Specialist": "Urologist", "Link": "https://www.practo.com/search/doctors?results_type=doctor&q=%5B%7B%22word%22%3A%22urologist%22%2C%22autocompleted%22%3Atrue%2C%22category%22%3A%22subspeciality%22%7D%5D&city"}
]

df1 = pd.DataFrame(specialists).reset_index()

df1

### By using the above links we scrap the data from each speciality

### Cocat all the csv files

df2 = pd.concat(
    map(pd.read_csv, ['Dentist.csv','Gynecologist.csv','Cardiologist.csv','Orthopedist.csv','Rheumatologist.csv','chiropractor.csv','bariatric_surgeon.csv','Pulmonologist.csv','Neurosurgeon.csv','Neurologist.csv','Gastroenterologist.csv','dietitian.csv','Psychiatrist.csv','Urologist.csv','Pediatrician.csv','Dermatologist.csv','Physiotherapist.csv','Infertility Specialist.csv','Ophthalmologist.csv']),ignore_index=True)
    
df2

df2['Speciality'].value_counts()

### Remove Dupicate Values in the DataSets
df2.drop_duplicates(inplace=True)

df2

### Data Cleaning

#### Null Handling & Remove Noise Values

# Years_of__Experience Column :- Extracting only numeric values
df2['Year_of_experience'] = df2['Year_of_experience'].str.extract('(\d+)')

# dp_score column :- Extracting only numeric values
df2['dp_score'] = df2['dp_score'].str.extract('(\d+)')

# npv column :- Extracting only numeric values
df2['npv'] = df2['npv'].str.extract('(\d+)')

# consultation_fee column :- Extracting only numeric values
df2['consultation_fee'] = df2['consultation_fee'].str.extract('(\d+)')

df2.head()

# Data Cleaning 
df2.isnull().sum()

df2.dropna(inplace=True)

# Check Shape of Datasets
df2.shape

for i in df2.columns:
    print(i,df2[i].sort_values().unique(),'\n',sep= '\n')
    
#### Change the DataTypes
df2.dtypes

# Converting to numeric column
for i in df2.columns:
    df2[i] = pd.to_numeric(df2[i] , errors = 'ignore')
    
df2.dtypes

#### In the data sets Location column and city column both have city name which is not generally required. so here we will split the column using ',' delimeter and then remove the column contain city name.
df2[['Location', 'b']] = df2.Location.str.split(",", expand = True)
df2.drop('b' , axis=1 , inplace=True)
df2

#### Separate the Numerical Columns and Categorical Columns
num=[]
cat=[]
for i in df2.columns:
    if df2[i].dtype=="O":
        cat.append(i)
    else:
        num.append(i)
num
cat
cat.remove('Name')

### Exploratory data analysis (EDA)
# Staststical Summary of Datasets
df2.describe()

for i in cat:
    print(df2[i].value_counts())
    
### Number of Doctors in Each City
#This will show numbers of doctors in each city
num_doc_city= df2.groupby(['City'],as_index=False)['Name'].count().sort_values(by='Name',ascending=False)
num_doc_city

##### From the above tables we clearly seen that Bangalore has highest number of doctors as compared to Delhi and Mumbai.
plt.figure(figsize=(8,4))
ax=sns.barplot(x='City',y='Name',data=num_doc_city)

# Add labels and title
plt.xlabel('City')
plt.ylabel('Number of doctors')
plt.title('Number of doctors by each city')

# Add values to the bars
for i, value in enumerate(num_doc_city['Name']):
    plt.text(i, value, str(value))
    
# Show the chart
plt.show()

### Count of doctors in each speciality
#This shows number of doctors in each speciality
num_doc_speciality= df2.groupby(['Speciality'],as_index=False)['Name'].count().sort_values(by='Name',ascending=False)
num_doc_speciality

### Number of doctors per speciality in each city

# this will show number of doctors of each speciality in each city

num_doc_speciality_city= df2.groupby(by=['City','Speciality'])['Name'].count().reset_index().sort_values(by=['City','Name'],ascending=[True,False])
num_doc_speciality_city

##### It is clearly seen that from above table in each city the number of dentist are more as compared to other speciality.

### Number of doctors in each Speciality having Highest number of Years of Experience
#This shows Top 10 number of doctors having Highest number  of Experience
num_doc_speciality_city= df2.groupby(by=['Year_of_experience','Speciality'])['Name'].count().reset_index().sort_values(by=['Year_of_experience','Name'],ascending=[True,False])
num_doc_speciality_city

### Speciality wise Fees Analysis
# To findout min,max and average consultaion fee charged by each speciality
speciality_fee= {'consultation_fee':['min','max','median']}
sp=df2.groupby(by=['Speciality']).agg(speciality_fee).reset_index()
sp

##### From above table we found out Neurosurgeon and Ophtalmologist speciality charges high consultaion fees and almost free consultaion chargres speciality will be Dentist, Dermatologist, Gynecologist/obstetrician, Infertility Specialist, Physiotherapist and dietitian/nutritionist
plt.figure(figsize=(16,7))  #graph size

plt.barh(num_doc_speciality['Speciality'],num_doc_speciality['Name'],color='g',label='No. of doctors',linestyle=':',linewidth=2)#assining x-y axis and other aesthetic
plt.title('Number of Doctors in different specialities',fontsize=30,color='r',weight='bold',loc='center')
plt.ylabel('Speciality',fontsize=20,weight='bold') #y-axis title
plt.xticks(fontsize = 12,weight='bold')
plt.yticks(fontsize = 12,weight='bold')

plt.grid(False) #adding gridlines
plt.legend(loc='upper left', title='Metric',fontsize=15,bbox_to_anchor=(1.07, 1),borderaxespad=0)

plt.show #print graph

#### From above bar chart we clearly seen that Most of the doctors are Dentist and least Chiropractor

# Doughnut chart
plt.figure(figsize=(10,5))  #graph size

explode = [0.1,0.1,0.1,0.1,0.1,0.1] # To slice the perticuler section
colors = ["c", 'b','r','y','g','m'] # Color of each section
textprops = {"fontsize":15} # Font size of text in pie chart

plt.pie(num_doc_city['Name'], # Values
        labels = num_doc_city['City'], # Labels for each sections
        colors =colors, # Color of each section
        autopct = "%0.2f%%", # Show data in persentage for with 2 decimal point
        shadow = True, # Showing shadow of pie chart
        radius = 1.4, # Radius to increase or decrease the size of pie chart 
       startangle = 270, # Start angle of first section
        textprops =textprops)

plt.pie([1],colors=['w'],radius=0.5)

plt.show #print graph
#### From the above Doughnut chart we clearly seen that the city like Bangalore has more percentages of doctors

## Correlation between the Variables By using Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df2.corr(),annot=True)
plt.title('Correlation-Heat Map',fontsize=30,color='b',weight='bold',loc='center')

df2['Year_of_experience'].value_counts()

### KDE Plot
#shows distribution of  year of experience
plt.figure(figsize=(4, 4))
plt.title('KDE Plot',fontsize=30,weight='bold',loc='center')
sns.kdeplot(df2['Year_of_experience'])

##### From above kde plot we clearly seen that Most of the doctors belongs to the range of 13 to 15 years of experience

### Count of Top 10 doctors  acoording to their location and city

# This table shows number of doctors according to their location in that city
num_of_location_city=df2.groupby(by=['City','Location'])['Name'].count().reset_index().sort_values(by=['City','Name'],ascending=[True,False])
num_of_location_city

# This table shows top 10 doctors accroding to their location and city
top_10=num_of_location_city.nlargest(10, columns=['Name'], keep='all')
top_10

#### The location like saket in delhi having most number of doctor are there

plt.figure(figsize=(12, 6))
# Create bar chart using seaborn
ax = sns.barplot(x='Name', y='Location', hue='City', data=top_10)

# Add labels and title
plt.xlabel('Number of doctors')
plt.ylabel('Location')
plt.title('Top 10 doctors by location and city')

# Add legend
ax.legend(loc='upper right')

# Add value labels to the bars
for i in ax.containers:
    ax.bar_label(i, label_type='edge')
    
# Show the chart
plt.show()

### Top 10 Degree
top_degree=df2['Degree'].value_counts().reset_index().head(10)
top_degree

##### In this table we found out most common degress  will be BDS
plt.figure(figsize=(8,4))
ax=sns.barplot(x='index',y='Degree',data=top_degree)

# Add labels and title
plt.xlabel('Degree')
plt.ylabel('Number of degree')
plt.title('Number of Degree in Speciality')

# Add values to the bars
for i, value in enumerate(top_degree['Degree']):
    plt.text(i, value, str(value))
    
# Rotate x-axis labels
plt.xticks(rotation=90)

# Show the chart
plt.show()

### Doctors having maximum number of specialization
doc_spec=df2['Name'].value_counts().reset_index().head(10)
doc_spec
top_10_bangalore = df2.loc[df2['City'] == 'Bangalore','Speciality'].value_counts().nlargest(10)
plt.figure(figsize=(8,6))
plt.bar(top_10_bangalore.index, top_10_bangalore.values)
plt.title('Top 10 Speciality in Bangalore')
plt.xticks(rotation=90)
plt.show()
top_10_Delhi = df2.loc[df2['City'] == 'Delhi','Speciality'].value_counts().nlargest(10)
plt.figure(figsize=(8,6))
plt.bar(top_10_Delhi.index, top_10_Delhi.values)
plt.title('Top 10 Speciality in Delhi')
plt.xticks(rotation=90)
plt.show()
top_10_Mumbai = df2.loc[df2['City'] == 'Mumbai','Speciality'].value_counts().nlargest(10)
plt.figure(figsize=(8,6))
plt.bar(top_10_Mumbai.index, top_10_Mumbai.values)
plt.title('Top 10 Speciality in Mumbai')
plt.xticks(rotation=90)
plt.show()
df2.drop('Name',axis=1, inplace=True)
df2

## Outlier Analysis
for i in num:
    plt.figure()
    sns.boxplot(y=i,data=df2)
for i in num:
    q1=df2[i].quantile(0.25)
    q3=df2[i].quantile(0.75)
    iqr=q3-q1
    ll=q1-1.5*iqr
    ul=q3+1.5*iqr
    df2=df2[(df2[i]>ll) & (df2[i]<ul)]
    plt.figure()
    sns.boxplot(y=i,data=df2)
    
df2.shape

## Data Preprocessing
from sklearn.preprocessing import LabelEncoder

# LabelEncoding
le=LabelEncoder()
col=['Degree','Location']
for j in col:
    df2[j]=le.fit_transform(df2[j])
df2.head()
#OneHotEncoding
df2_new = pd.get_dummies(df2, columns = ['Speciality','City'])
df2_new .head()

### Extract the independent and dependent variable

# Independent Variable
X=df2_new.drop(['consultation_fee'], axis=1).values

# Dependent Variable
Y=df2_new['consultation_fee'].values

### Split the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
from sklearn.preprocessing import StandardScaler

### Scaling the data
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

### Linear Regression Model
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, Y_train)
Y_pred=reg.predict(X_test)
Y_pred

### Evaluating the model using MSE, RMSE and R square
from sklearn import metrics
# Mean Square Error(MSE)
MSE=metrics.mean_squared_error(Y_test, Y_pred)
print('MSE =',MSE)
# Root mean square error(RSME)
RMSE=np.sqrt(MSE)
print('RMSE =',RMSE)
# Coefficient of determination or R-squared
R2=metrics.r2_score(Y_test,Y_pred)
print('R-squared =',R2)
print('Coefficients:',reg.coef_)
print('Intercept:',reg.intercept_)


