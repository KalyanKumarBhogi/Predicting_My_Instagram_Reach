# Predicting_My_Instagram_Reach
Instagram is one of the most popular social media applications today. People using Instagram professionally are using it for promoting their business, building a portfolio, blogging, and creating various kinds of content. As Instagram is a popular application used by millions of people with different niches, Instagram keeps changing to make itself better for the content creators and the users. But as this keeps changing, it affects the reach of our posts that affects us in the long run. So if a content creator wants to do well on Instagram in the long run, they have to look at the data of their Instagram reach. This project( Instagram Reach Analysis using Python), which will help content creators to understand how to adapt to the changes in Instagram in the long run.

# Instagram Reach Analysis
I have been researching Instagram reach for a long time now. Every time I post on my Instagram account, I collect data on how well the post reach after a week. That helps in understanding how Instagram’s algorithm is working. If you want to analyze the reach of your Instagram account, you have to collect your data manually as there are some APIs, but they don’t work well. So it’s better to collect your Instagram data manually.

# Instagram Reach Analysis using Python
Now let’s start the task of analyzing the reach of my Instagram account by importing the necessary Python libraries and the dataset:

import pandas as pd </p>
import numpy as np </p>
import matplotlib.pyplot as plt </p>
import seaborn as sns </p>
import plotly.express as px </p>
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator </p>
from sklearn.model_selection import train_test_split </p>
from sklearn.linear_model import PassiveAggressiveRegressor </p>

data = pd.read_csv("Instagram.csv", encoding = 'latin1') </p>
print(data.head()) </p>

Before starting everything, let’s have a look at whether this dataset contains any null values or not: </p>
data.isnull().sum() </p>

# Analyzing Instagram Reach
Now let’s start with analyzing the reach of my Instagram posts. I will first have a look at the distribution of impressions I have received from home:  </p>

plt.figure(figsize=(10, 8)) </p>
plt.style.use('fivethirtyeight') </p>
plt.title("Distribution of Impressions From Home") </p>
sns.distplot(data['From Home']) </p>
plt.show() </p>

The impressions I get from the home section on Instagram shows how much my posts reach my followers. Looking at the impressions from home, I can say it’s hard to reach all my followers daily. Now let’s have a look at the distribution of the impressions I received from hashtags:

plt.figure(figsize=(10, 8))  </p>
plt.title("Distribution of Impressions From Hashtags")  </p>
sns.distplot(data['From Hashtags'])  </p>
plt.show()   </p>

Hashtags are tools we use to categorize our posts on Instagram so that we can reach more people based on the kind of content we are creating. Looking at hashtag impressions shows that not all posts can be reached using hashtags, but many new users can be reached from hashtags. Now let’s have a look at the distribution of impressions I have received from the explore section of Instagram:

plt.figure(figsize=(10, 8))   </p>
plt.title("Distribution of Impressions From Explore")  </p>
sns.distplot(data['From Explore'])  </p>
plt.show()   </p>

The explore section of Instagram is the recommendation system of Instagram. It recommends posts to the users based on their preferences and interests. By looking at the impressions I have received from the explore section, I can say that Instagram does not recommend our posts much to the users. Some posts have received a good reach from the explore section, but it’s still very low compared to the reach I receive from hashtags.

Now let’s have a look at the percentage of impressions I get from various sources on Instagram:

home = data["From Home"].sum()   </p>
hashtags = data["From Hashtags"].sum()   </p>
explore = data["From Explore"].sum()  </p>
other = data["From Other"].sum()    </p>

labels = ['From Home','From Hashtags','From Explore','Other']   </p>
values = [home, hashtags, explore, other]    </p>

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)   </p>
fig.show()    </p>

So the above donut plot shows that almost 50 per cent of the reach is from my followers, 38.1 per cent is from hashtags, 9.14 per cent is from the explore section, and 3.01 per cent is from other sources.

# Analyzing Relationships
Now let’s analyze relationships to find the most important factors of our Instagram reach. It will also help us in understanding how the Instagram algorithm works.

Let’s have a look at the relationship between the number of likes and the number of impressions on my Instagram posts:

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")   </p>
figure.show()    </p>

There is a linear relationship between the number of likes and the reach I got on Instagram. Now let’s see the relationship between the number of comments and the number of impressions on my Instagram posts:

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")  </p>
figure.show()  </p>

It looks like the number of comments we get on a post doesn’t affect its reach. Now let’s have a look at the relationship between the number of shares and the number of impressions:

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title = "Relationship Between Shares and Total Impressions")  </p>
figure.show()    </p>

A more number of shares will result in a higher reach, but shares don’t affect the reach of a post as much as likes do. Now let’s have a look at the relationship between the number of saves and the number of impressions:

figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title = "Relationship Between Post Saves and Total Impressions")  </p>
figure.show()   </p>

There is a linear relationship between the number of times my post is saved and the reach of my Instagram post. Now let’s have a look at the correlation of all the columns with the Impressions column:
correlation = data.corr()  </p>
print(correlation["Impressions"].sort_values(ascending=False))  </p>

So we can say that more likes and saves will help you get more reach on Instagram. The higher number of shares will also help you get more reach, but a low number of shares will not affect your reach either.

# Analyzing Conversion Rate
In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. The formula that you can use to calculate conversion rate is (Follows/Profile Visits) * 100. Now let’s have a look at the conversation rate of my Instagram account:

So the conversation rate of my Instagram account is 31% which sounds like a very good conversation rate. Let’s have a look at the relationship between the total profile visits and the number of followers gained from all profile visits:

figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")  </p>
figure.show()  </p>

# Instagram Reach Prediction Model
Now in this section, I will train a machine learning model to predict the reach of an Instagram post. Let’s split the data into training and test sets before training the model:

x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])  </p>
y = np.array(data["Impressions"])  </p>
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)  </p>
                                                
Now here’s is how we can train a machine learning model to predict the reach of an Instagram post using Python:

 model = PassiveAggressiveRegressor()  </p>
model.fit(xtrain, ytrain)  </p>
model.score(xtest, ytest) </p>

Now let’s predict the reach of an Instagram post by giving inputs to the machine learning model: </p>

#Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]  </p>
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])  </p>
model.predict(features) </p>

# Summary </p>
So this is how you can analyze and predict the reach of Instagram posts with machine learning using Python. If a content creator wants to do well on Instagram in a long run, they have to look at the data of their Instagram reach. That is where the use of Data Science in social media comes in.
