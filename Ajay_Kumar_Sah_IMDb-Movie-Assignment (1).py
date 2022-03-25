#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Filtering out the warnings

import warnings

warnings.filterwarnings('ignore')


# In[2]:


# Importing the required libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # <font color = blue> IMDb Movie Assignment </font>
# 
# You have the data for the 100 top-rated movies from the past decade along with various pieces of information about the movie, its actors, and the voters who have rated these movies online. In this assignment, you will try to find some interesting insights into these movies and their voters, using Python.

# ##  Task 1: Reading the data

# - ### Subtask 1.1: Read the Movies Data.
# 
# Read the movies data file provided and store it in a dataframe `movies`.

# In[18]:


# Read the csv file using 'read_csv'. Please write your dataset location here.

df1 = pd.read_csv("Movie+Assignment+Data.csv")
df1.head()


# - ###  Subtask 1.2: Inspect the Dataframe
# 
# Inspect the dataframe for dimensions, null-values, and summary of different numeric columns.

# In[17]:


# Check the number of rows and columns in the dataframe

rows = len(df1.axes[0])

cols = len(df1.axes[1])

print("No of rows", rows)
print("No of Cols", cols)


# In[20]:


# Check the column-wise info of the dataframe

df1.info()


# In[21]:


# Check the summary for the numeric columns 

df1.describe()


# ## Task 2: Data Analysis
# 
# Now that we have loaded the dataset and inspected it, we see that most of the data is in place. As of now, no data cleaning is required, so let's start with some data manipulation, analysis, and visualisation to get various insights about the data. 

# -  ###  Subtask 2.1: Reduce those Digits!
# 
# These numbers in the `budget` and `gross` are too big, compromising its readability. Let's convert the unit of the `budget` and `gross` columns from `$` to `million $` first.

# In[22]:


# Divide the 'gross' and 'budget' columns by 1000000 to convert '$' to 'million $'

df1['Gross'] = df1['Gross'].div(1000000)
df1['budget'] = df1['budget'].div(1000000)

df1.head()


# -  ###  Subtask 2.2: Let's Talk Profit!
# 
#     1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
#     2. Sort the dataframe using the `profit` column as reference.
#     3. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`.
#     4. Plot a scatter or a joint plot between the columns `budget` and `profit` and write a few words on what you observed.
#     5. Extract the movies with a negative profit and store them in a new dataframe - `neg_profit`

# In[23]:


# Create the new column named 'profit' by subtracting the 'budget' column from the 'gross' column

df1['Profit'] = df1['Gross'] - df1['budget']

df1.head()


# In[24]:


# Sort the dataframe with the 'profit' column as reference using the 'sort_values' function. Make sure to set the argument
#'ascending' to 'False'

df1.sort_values(by=['Profit'], inplace=True, ascending=False)

df1.head()


# In[85]:


# Get the top 10 profitable movies by using position based indexing. Specify the rows till 10 (0-9)

df1[:10]


# In[50]:


#Plot profit vs budget

import matplotlib.pyplot as plt

profit = df1.Profit
budget = df1.budget

plt.scatter(profit,budget, alpha = 0.9, s = 50)
plt.title("Profit versus Budget \n", fontdict={'fontsize':20, 'fontweight':5, 'color':'Green'})

plt.xlabel("Budget", fontdict={'fontsize':12, 'fontweight':10,'color':'Green'})
plt.ylabel("Profit", fontdict={'fontsize':12, 'fontweight':10, 'color':'Red'})

plt.show()


# The dataset contains the 100 best performing movies from the year 2010 to 2016. However, the scatter plot tells a different story. You can notice that there are some movies with negative profit. Although good movies do incur losses, but there appear to be quite a few movie with losses. What can be the reason behind this? Lets have a closer look at this by finding the movies with negative profit.

# In[86]:


#Find the movies with negative profit

df1[(df1.Profit < 0)]


# **`Checkpoint 1:`** Can you spot the movie `Tangled` in the dataset? You may be aware of the movie 'Tangled'. Although its one of the highest grossing movies of all time, it has negative profit as per this result. If you cross check the gross values of this movie (link: https://www.imdb.com/title/tt0398286/), you can see that the gross in the dataset accounts only for the domestic gross and not the worldwide gross. This is true for may other movies also in the list.

# - ### Subtask 2.3: The General Audience and the Critics
# 
# You might have noticed the column `MetaCritic` in this dataset. This is a very popular website where an average score is determined through the scores given by the top-rated critics. Second, you also have another column `IMDb_rating` which tells you the IMDb rating of a movie. This rating is determined by taking the average of hundred-thousands of ratings from the general audience. 
# 
# As a part of this subtask, you are required to find out the highest rated movies which have been liked by critics and audiences alike.
# 1. Firstly you will notice that the `MetaCritic` score is on a scale of `100` whereas the `IMDb_rating` is on a scale of 10. First convert the `MetaCritic` column to a scale of 10.
# 2. Now, to find out the movies which have been liked by both critics and audiences alike and also have a high rating overall, you need to -
#     - Create a new column `Avg_rating` which will have the average of the `MetaCritic` and `Rating` columns
#     - Retain only the movies in which the absolute difference(using abs() function) between the `IMDb_rating` and `Metacritic` columns is less than 0.5. Refer to this link to know how abs() funtion works - https://www.geeksforgeeks.org/abs-in-python/ .
#     - Sort these values in a descending order of `Avg_rating` and retain only the movies with a rating equal to or greater than `8` and store these movies in a new dataframe `UniversalAcclaim`.
#     

# In[87]:


# Change the scale of MetaCritic

df1['MetaCritic'] = df1['MetaCritic'].div(10)


# In[89]:


# Find the average ratings

df1['Avg_rating'] = df1[['MetaCritic', 'IMDb_rating']].mean(axis=1)


# In[99]:


#Sort in descending order of average rating

df1.sort_values(by=['Avg_rating'], inplace=True, ascending=False)
df1


# In[149]:


# Find the movies with metacritic-Imdb rating < 0.5 and also with an average rating of >= 8 (sorted in descending order)

UniversalAcclaim = df1[(abs(df1.MetaCritic - df1.IMDb_rating) < 0.5) & (df1.Avg_rating >= 8)]

UniversalAcclaim


# **`Checkpoint 2:`** Can you spot a `Star Wars` movie in your final dataset?

# - ### Subtask 2.4: Find the Most Popular Trios - I
# 
# You're a producer looking to make a blockbuster movie. There will primarily be three lead roles in your movie and you wish to cast the most popular actors for it. Now, since you don't want to take a risk, you will cast a trio which has already acted in together in a movie before. The metric that you've chosen to check the popularity is the Facebook likes of each of these actors.
# 
# The dataframe has three columns to help you out for the same, viz. `actor_1_facebook_likes`, `actor_2_facebook_likes`, and `actor_3_facebook_likes`. Your objective is to find the trios which has the most number of Facebook likes combined. That is, the sum of `actor_1_facebook_likes`, `actor_2_facebook_likes` and `actor_3_facebook_likes` should be maximum.
# Find out the top 5 popular trios, and output their names in a list.
# 

# In[161]:


# Write your code here

df1['Actor_Facebook_likes'] = df1[['actor_1_facebook_likes', 'actor_2_facebook_likes','actor_3_facebook_likes']].sum(axis=1)

df1.sort_values(by=['Actor_Facebook_likes'], inplace=True, ascending=False)
df1.head(5)


# - ### Subtask 2.5: Find the Most Popular Trios - II
# 
# In the previous subtask you found the popular trio based on the total number of facebook likes. Let's add a small condition to it and make sure that all three actors are popular. The condition is **none of the three actors' Facebook likes should be less than half of the other two**. For example, the following is a valid combo:
# - actor_1_facebook_likes: 70000
# - actor_2_facebook_likes: 40000
# - actor_3_facebook_likes: 50000
# 
# But the below one is not:
# - actor_1_facebook_likes: 70000
# - actor_2_facebook_likes: 40000
# - actor_3_facebook_likes: 30000
# 
# since in this case, `actor_3_facebook_likes` is 30000, which is less than half of `actor_1_facebook_likes`.
# 
# Having this condition ensures that you aren't getting any unpopular actor in your trio (since the total likes calculated in the previous question doesn't tell anything about the individual popularities of each actor in the trio.).
# 
# You can do a manual inspection of the top 5 popular trios you have found in the previous subtask and check how many of those trios satisfy this condition. Also, which is the most popular trio after applying the condition above? Write your answers in the markdown cell provided below.

# **Write your answers below.**
# 
# - **`No. of trios that satisfy the above condition:`** (your answer here)
# 
# - **`Most popular trio after applying the condition:`** (your answer here)

# **`Optional:`** Even though you are finding this out by a natural inspection of the dataframe, can you also achieve this through some *if-else* statements to incorporate this. You can try this out on your own time after you are done with the assignment.

# In[ ]:


# Your answer here (optional and not graded)


# - ### Subtask 2.6: Runtime Analysis
# 
# There is a column named `Runtime` in the dataframe which primarily shows the length of the movie. It might be intersting to see how this variable this distributed. Plot a `histogram` or `distplot` of seaborn to find the `Runtime` range most of the movies fall into.

# In[490]:


# Runtime histogram/density plot

sns.histplot(data = df1 ,x = 'Runtime')

sns.distplot(df1.Runtime)


# **`Checkpoint 3:`** Most of the movies appear to be sharply 2 hour-long.

# - ### Subtask 2.7: R-Rated Movies
# 
# Although R rated movies are restricted movies for the under 18 age group, still there are vote counts from that age group. Among all the R rated movies that have been voted by the under-18 age group, find the top 10 movies that have the highest number of votes i.e.`CVotesU18` from the `movies` dataframe. Store these in a dataframe named `PopularR`.

# In[206]:


# Write your code here

PopularR = df1.sort_values("CVotesU18", ascending=False)
PopularR.head(10)


# **`Checkpoint 4:`** Are these kids watching `Deadpool` a lot?

#  

# ## Task 3 : Demographic analysis
# 
# If you take a look at the last columns in the dataframe, most of these are related to demographics of the voters (in the last subtask, i.e., 2.8, you made use one of these columns - CVotesU18). We also have three genre columns indicating the genres of a particular movie. We will extensively use these columns for the third and the final stage of our assignment wherein we will analyse the voters across all demographics and also see how these vary across various genres. So without further ado, let's get started with `demographic analysis`.

# -  ###  Subtask 3.1 Combine the Dataframe by Genres
# 
# There are 3 columns in the dataframe - `genre_1`, `genre_2`, and `genre_3`. As a part of this subtask, you need to aggregate a few values over these 3 columns. 
# 1. First create a new dataframe `df_by_genre` that contains `genre_1`, `genre_2`, and `genre_3` and all the columns related to **CVotes/Votes** from the `movies` data frame. There are 47 columns to be extracted in total.
# 2. Now, Add a column called `cnt` to the dataframe `df_by_genre` and initialize it to one. You will realise the use of this column by the end of this subtask.
# 3. First group the dataframe `df_by_genre` by `genre_1` and find the sum of all the numeric columns such as `cnt`, columns related to CVotes and Votes columns and store it in a dataframe `df_by_g1`.
# 4. Perform the same operation for `genre_2` and `genre_3` and store it dataframes `df_by_g2` and `df_by_g3` respectively. 
# 5. Now that you have 3 dataframes performed by grouping over `genre_1`, `genre_2`, and `genre_3` separately, it's time to combine them. For this, add the three dataframes and store it in a new dataframe `df_add`, so that the corresponding values of Votes/CVotes get added for each genre.There is a function called `add()` in pandas which lets you do this. You can refer to this link to see how this function works. https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.add.html
# 6. The column `cnt` on aggregation has basically kept the track of the number of occurences of each genre.Subset the genres that have atleast 10 movies into a new dataframe `genre_top10` based on the `cnt` column value.
# 7. Now, take the mean of all the numeric columns by dividing them with the column value `cnt` and store it back to the same dataframe. We will be using this dataframe for further analysis in this task unless it is explicitly mentioned to use the dataframe `movies`.
# 8. Since the number of votes can't be a fraction, type cast all the CVotes related columns to integers. Also, round off all the Votes related columns upto two digits after the decimal point.
# 

# In[248]:


# Create the dataframe df_by_genre

df_by_genre = df1.iloc[:,[11,12,13,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]]
df_by_genre.info()


# In[251]:


# Create a column cnt and initialize it to 1

df_by_genre["cnt"] = 1
df_by_genre


# In[574]:


# Group the movies by individual genres

df_by_g1 = df_by_genre.groupby('genre_1').sum()
df_by_g1


# In[578]:


# Add the grouped data frames and store it in a new data frame

df_by_g2 = df_by_genre.groupby('genre_2').sum()
df_by_g2

df_by_g3 = df_by_genre.groupby('genre_3').sum()
df_by_g3

df_add = df_by_g1.add(df_by_g2, fill_value=0)
df_add = df_add.add(df_by_g3,fill_value=0)
df_add


# In[585]:


# Extract genres with atleast 10 occurences

genre_top10 = df_add[(df_add.cnt >= 10)]
genre_top10


# In[588]:


# Take the mean for every column by dividing with cnt 

genre_top10.iloc[:,0:].div(genre_top10.cnt, axis = 0).mean(axis=0)


# In[589]:


# Rounding off the columns of Votes to two decimals

genre_top10[['VotesM','VotesF','VotesU18','VotesU18M','VotesU18F','Votes1829','Votes1829M','Votes1829F','Votes3044','Votes3044M','Votes3044F','Votes45A','Votes45AM','Votes45AF','Votes1000','VotesUS','VotesnUS']].round(2)


# In[590]:


# Converting CVotes to int type

genre_top10[["CVotes10","CVotes09","CVotes08","CVotes07","CVotes06","CVotes05","CVotes04","CVotes03","CVotes02","CVotes01","CVotesMale","CVotesFemale","CVotesU18","CVotesU18M","CVotesU18F","CVotes1829","CVotes1829M","CVotes1829F","CVotes3044","CVotes3044M","CVotes3044F","CVotes45A","CVotes45AM","CVotes45AF","CVotes1000","CVotesUS","CVotesnUS"]].astype(int)


# If you take a look at the final dataframe that you have gotten, you will see that you now have the complete information about all the demographic (Votes- and CVotes-related) columns across the top 10 genres. We can use this dataset to extract exciting insights about the voters!

# -  ###  Subtask 3.2: Genre Counts!
# 
# Now let's derive some insights from this data frame. Make a bar chart plotting different genres vs cnt using seaborn.

# In[594]:


# Countplot for genres

genre_top10_copy = genre_top10.copy(deep=True)
genre_top10_copy['Genre'] = genre_top10_copy.index
genre_top10_copy
sns.barplot(data = genre_top10_copy, x = 'Genre', y = 'cnt')
plt.xticks(rotation=90)
plt.show()


# **`Checkpoint 5:`** Is the bar for `Drama` the tallest?

# -  ###  Subtask 3.3: Gender and Genre
# 
# If you have closely looked at the Votes- and CVotes-related columns, you might have noticed the suffixes `F` and `M` indicating Female and Male. Since we have the vote counts for both males and females, across various age groups, let's now see how the popularity of genres vary between the two genders in the dataframe. 
# 
# 1. Make the first heatmap to see how the average number of votes of males is varying across the genres. Use seaborn heatmap for this analysis. The X-axis should contain the four age-groups for males, i.e., `CVotesU18M`,`CVotes1829M`, `CVotes3044M`, and `CVotes45AM`. The Y-axis will have the genres and the annotation in the heatmap tell the average number of votes for that age-male group. 
# 
# 2. Make the second heatmap to see how the average number of votes of females is varying across the genres. Use seaborn heatmap for this analysis. The X-axis should contain the four age-groups for females, i.e., `CVotesU18F`,`CVotes1829F`, `CVotes3044F`, and `CVotes45AF`. The Y-axis will have the genres and the annotation in the heatmap tell the average number of votes for that age-female group. 
# 
# 3. Make sure that you plot these heatmaps side by side using `subplots` so that you can easily compare the two genders and derive insights.
# 
# 4. Write your any three inferences from this plot. You can make use of the previous bar plot also here for better insights.
# Refer to this link- https://seaborn.pydata.org/generated/seaborn.heatmap.html. You might have to plot something similar to the fifth chart in this page (You have to plot two such heatmaps side by side).
# 
# 5. Repeat subtasks 1 to 4, but now instead of taking the CVotes-related columns, you need to do the same process for the Votes-related columns. These heatmaps will show you how the two genders have rated movies across various genres.
# 
# You might need the below link for formatting your heatmap.
# https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
# 
# -  Note : Use `genre_top10` dataframe for this subtask

# In[595]:


# 1st set of heat maps for CVotes-related columns

sns.heatmap(data=genre_top10[['CVotesU18M','CVotes1829M','CVotes3044M','CVotes45AM']])
plt.show()


# **`Inferences:`** A few inferences that can be seen from the heatmap above is that males have voted more than females, and Sci-Fi appears to be most popular among the 18-29 age group irrespective of their gender. What more can you infer from the two heatmaps that you have plotted? Write your three inferences/observations below:
# - Inference 1: Under 18 age group has voted the least
# - Inference 2: Drama  is the most popular movie
# - Inference 3: Among the 30-44 age group is the 2nd most voted

# In[596]:


# 2nd set of heat maps for Votes-related columns

sns.heatmap(data=genre_top10[['CVotesU18F','CVotes1829F', 'CVotes3044F','CVotes45AF']])
plt.show()


# **`Inferences:`** Sci-Fi appears to be the highest rated genre in the age group of U18 for both males and females. Also, females in this age group have rated it a bit higher than the males in the same age group. What more can you infer from the two heatmaps that you have plotted? Write your three inferences/observations below:
# - Inference 1: 18-29 age group are the most voted
# - Inference 2: Sci-Fi is the most highest rated in both the gender
# - Inference 3: under 18 is the leasted voted

# -  ###  Subtask 3.4: US vs non-US Cross Analysis
# 
# The dataset contains both the US and non-US movies. Let's analyse how both the US and the non-US voters have responded to the US and the non-US movies.
# 
# 1. Create a column `IFUS` in the dataframe `movies`. The column `IFUS` should contain the value "USA" if the `Country` of the movie is "USA". For all other countries other than the USA, `IFUS` should contain the value `non-USA`.
# 
# 
# 2. Now make a boxplot that shows how the number of votes from the US people i.e. `CVotesUS` is varying for the US and non-US movies. Make use of the column `IFUS` to make this plot. Similarly, make another subplot that shows how non US voters have voted for the US and non-US movies by plotting `CVotesnUS` for both the US and non-US movies. Write any of your two inferences/observations from these plots.
# 
# 
# 3. Again do a similar analysis but with the ratings. Make a boxplot that shows how the ratings from the US people i.e. `VotesUS` is varying for the US and non-US movies. Similarly, make another subplot that shows how `VotesnUS` is varying for the US and non-US movies. Write any of your two inferences/observations from these plots.
# 
# Note : Use `movies` dataframe for this subtask. Make use of this documention to format your boxplot - https://seaborn.pydata.org/generated/seaborn.boxplot.html

# In[600]:


# Creating IFUS column

df1['IFUS'] = ['USA' if x == 'USA' else 'non-USA' for x in df1['Country']]
df1


# In[615]:


# Box plot - 1: CVotesUS(y) vs IFUS(x)

sns.boxplot(x='IFUS', y='CVotesUS', data = df1)
plt.show()

sns.boxplot(x='IFUS', y='CVotesnUS', data = df1)
plt.show()


# **`Inferences:`** Write your two inferences/observations below:
# - Inference 1: Median is nearly the same
# - Inference 2: There are some votes are which are exceptionally higher in USA

# In[616]:


# Box plot - 2: VotesUS(y) vs IFUS(x)

sns.boxplot(x='IFUS', y='VotesUS', data = df1)
plt.show()

sns.boxplot(x='IFUS', y='VotesnUS', data = df1)
plt.show()


# **`Inferences:`** Write your two inferences/observations below:
# - Inference 1: There are only few outliers of votes in USA
# - Inference 2: Non-USA countries dont have any outliers in votes

# -  ###  Subtask 3.5:  Top 1000 Voters Vs Genres
# 
# You might have also observed the column `CVotes1000`. This column represents the top 1000 voters on IMDb and gives the count for the number of these voters who have voted for a particular movie. Let's see how these top 1000 voters have voted across the genres. 
# 
# 1. Sort the dataframe genre_top10 based on the value of `CVotes1000`in a descending order.
# 
# 2. Make a seaborn barplot for `genre` vs `CVotes1000`.
# 
# 3. Write your inferences. You can also try to relate it with the heatmaps you did in the previous subtasks.
# 
# 
# 

# In[617]:


# Sorting by CVotes1000

genre_top10.sort_values("CVotes1000", ascending=False)


# In[622]:


# Bar plot

genre_top10_copy1 = genre_top10.copy(deep=True)
genre_top10_copy1['Genre'] = genre_top10_copy1.index
genre_top10_copy
sns.barplot(data = genre_top10_copy1, y = 'Genre', x = 'CVotes1000')
plt.xticks(rotation=90)
plt.show()


# **`Inferences:`** Write your inferences/observations here.
# --> Drama is most watchable movies

# **`Checkpoint 6:`** The genre `Romance` seems to be most unpopular among the top 1000 voters.

# 
# 
# 

# With the above subtask, your assignment is over. In your free time, do explore the dataset further on your own and see what kind of other insights you can get across various other columns.
