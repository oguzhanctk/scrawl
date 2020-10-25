import pandas as pd #for data processing
import numpy as np #for n-dimensional array operations
import seaborn as sns #for data visualization
import matplotlib.pyplot as plt
from collections import Counter

#%%

#read .csv file
medianHouseholdIncome2015 = pd.read_csv("MedianHouseholdIncome2015.csv", encoding="ISO-8859-1")
percentagePeopleBelowPovertyLevel = pd.read_csv("PercentagePeopleBelowPovertyLevel.csv", encoding="ISO-8859-1") 
percentOver25CompletedHighSchool = pd.read_csv("PercentOver25CompletedHighSchool.csv", encoding="ISO-8859-1")
policeKillingsUS = pd.read_csv("PoliceKillingsUS.csv", encoding="ISO-8859-1")
shareRaceByCity = pd.read_csv("ShareRaceByCity.csv", encoding="ISO-8859-1")

#POVERTY RATE BY GIVEN STATES

#replace pointless values to float 0.0
percentagePeopleBelowPovertyLevel.poverty_rate.replace(["-"], 0.0, inplace=True)

#poverty_rate is not numerical we must convert all values to float
percentagePeopleBelowPovertyLevel.poverty_rate = percentagePeopleBelowPovertyLevel.poverty_rate.astype(float)

#%%

#take all unique states code -> find average for poverty_rate ->  sort new df by poverty_rate
unique_area_list = list(percentagePeopleBelowPovertyLevel["Geographic Area"].unique())
area_poverty_rate = []

for member in unique_area_list:
    x = percentagePeopleBelowPovertyLevel[percentagePeopleBelowPovertyLevel["Geographic Area"] == member]
    avg_poverty_rate = sum(x.poverty_rate) / len(x)
    area_poverty_rate.append(avg_poverty_rate)

data = pd.DataFrame({"State": unique_area_list, "poverty_rate" : area_poverty_rate})
data = data.sort_values(by="poverty_rate", ascending=False)

#visualization

plt.figure(figsize=(15,10))
sns.barplot(data=data, x="State", y="poverty_rate", edgecolor="white")
plt.title("Poverty Rate by States")
plt.xticks(rotation=45)
plt.xlabel("States")
plt.ylabel("Poverty Rate")


#%%
#MOST COMMON 15 NAME OR SURNAME OF KILLED PEOPLE

#get name and surname into seperate list and combine them 
temp = policeKillingsUS.name[policeKillingsUS.name != "TK TK"].str.split()
a,b = zip(*temp)
name_surname_list = a + b
most_common_name_surname = Counter(name_surname_list).most_common(20)

#n is list for names and r is list for rate -> get name and rate to lists and put them axises
n,r = zip(*most_common_name_surname)
n,r = list(n), list(r)

#visualization

plt.figure(figsize=(15,10))
sns.barplot(x=n, y=r, color="#49c98f", palette=sns.cubehelix_palette(len(n)))
plt.xlabel("Name or Surname of killed people")
plt.ylabel("Frequency")
plt.title("Most Common 15 Name or Surname of people who killed by police")

# %%
#PERCENTAGE OF POPULATION ABOVE 25 THAT HAS GRADUATED HIGH SCHOOL BY STATE

#"percent_completed_hs" column has "-" value and we should convert this to float 0.0 -> convert "percent_completed_hs" column to float64
percentOver25CompletedHighSchool.percent_completed_hs.replace(["-"], 0.0, inplace=True)
percentOver25CompletedHighSchool.percent_completed_hs = percentOver25CompletedHighSchool.percent_completed_hs.astype(float)

#get unique list of states names
unique_state_code = list(percentOver25CompletedHighSchool["Geographic Area"].unique())

#calculate the average rate of each state
state_rate_arr = []

for member in unique_state_code:
    temp = percentOver25CompletedHighSchool[percentOver25CompletedHighSchool["Geographic Area"] == member]
    avg = sum(temp.percent_completed_hs) / len(temp)
    state_rate_arr.append(avg)

result_data = pd.DataFrame({"states" : unique_state_code, "rate" : state_rate_arr}).sort_values(by="rate")

#visualization

plt.figure(figsize=(15,10))
sns.barplot(data=result_data, x="states", y="rate", palette=sns.cubehelix_palette(len(result_data)))
plt.xlabel("States")
plt.ylabel("Graduated Rate")
plt.xticks(rotation=60)
plt.title("PERCENTAGE OF POPULATION ABOVE 25 THAT HAS GRADUATED HIGH SCHOOL BY STATE")

# %%

#PERCENTAGE OF STATE'S POPULATION ACCORDING TO RACES

#remove pointless characters from lists

check = shareRaceByCity.isin(["-", "(X)", "(x)"]).any() #check pointless character existence
shareRaceByCity.replace("(X)", 0.0, inplace=True) #replace
shareRaceByCity.loc[:, ["share_white", "share_black", "share_native_american", "share_asian", "share_hispanic"]] = shareRaceByCity.loc[:, ["share_white", "share_black", "share_native_american", "share_asian", "share_hispanic"]].astype(float) #convert to float
state_list = list(shareRaceByCity["Geographic area"].unique())


#define five lists to keep floats by race
share_white = []
share_black = []
share_native_american = []
share_asian = []
share_hispanic = []

for member in state_list:
    x = shareRaceByCity[shareRaceByCity["Geographic area"] == member]
    share_white.append(sum(x.share_white) / len(x))
    share_black.append(sum(x.share_black) / len(x))
    share_native_american.append(sum(x.share_native_american) / len(x))
    share_asian.append(sum(x.share_asian) / len(x))
    share_hispanic.append(sum(x.share_hispanic) / len(x))

#visualization
plt.figure(figsize=(10,15))
sns.barplot(x=share_white, y=state_list, color="red", label="white", alpha=0.6)
sns.barplot(x=share_black, y=state_list, color="blue", label="black", alpha=0.6)
sns.barplot(x=share_native_american, y=state_list, color="green", label="native-american", alpha=0.6)
sns.barplot(x=share_asian, y=state_list, color="orange", label="asian", alpha=0.6)
sns.barplot(x=share_hispanic, y=state_list, color="cyan", label="hispanic", alpha=0.6)
plt.legend(loc="center left", frameon = True, bbox_to_anchor=(1, 0.5, ))



# %%

#HIGH SCHOOL GRADUATION RATE VS POVERTY RATE OF EACH STATE

#result_data = sorted graduation dataframe for each state, data = sorted poverty rate for each state 
result_data.rename(columns={"rate" : "graduation_rate"}, inplace=True)
result_data["graduation_rate"] = result_data["graduation_rate"] / max(result_data["graduation_rate"])
data["poverty_rate"] = data["poverty_rate"] / max(data["poverty_rate"])
combine_data = pd.concat([result_data, data["poverty_rate"]], axis=1)
combine_data.sort_values(by="poverty_rate", ascending=True, inplace=True)
area_list = list(combine_data.states.unique())

#visusalization

plt.subplots(figsize=(13,13))
sns.pointplot(data=combine_data, x="states", y="poverty_rate", color="red", label="poverty_rate")
sns.pointplot(data=combine_data, x="states", y="graduation_rate", color="lime", label="graduation_rate")
plt.xlabel("States")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.title("HIGH SCHOOL GRADUATION RATE VS POVERTY RATE OF EACH STATE")
plt.text(40, 0.3, "poverty_rate", color="red", fontsize=18)
plt.text(40, 0.35, "graduation_rate", color="lime", fontsize=18)
plt.grid()


# %%

#JOINT PLOT
#kind="scatter(default) | hex | kde | resid | reg"
j = sns.jointplot("poverty_rate", "graduation_rate", data=combine_data, kind="reg", size=7)

# %%

#Different types of plots in seaborn

#remove all rows that has NaN values -> uncomment any line and draw graph
policeKillingsUS.race.dropna(inplace=True)
sizes = policeKillingsUS.race.value_counts().values
labels = policeKillingsUS.race.value_counts().index
policeKillingsUS.race.value_counts()
explode = [0, 0, 0, 0, 0, 0]
temp_data = combine_data.append({"graduation_rate" : 0.2, "poverty_rate" : 0.1, "states" : "AQ"}, ignore_index=True)
temp_data.head()

x = policeKillingsUS.city.value_counts().index
y = policeKillingsUS.city.value_counts().values

# plt.figure(figsize=(15,10))
#1 -> plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
#2 -> sns.lmplot(x="poverty_rate", y="graduation_rate", data=combine_data)
#3 -> sns.kdeplot(combine_data.poverty_rate, combine_data.graduation_rate, shade=True, cut=3)
sns.violinplot(data=temp_data, inner="points")
#5 -> sns.heatmap(combine_data.corr(), annot=True, linewidths=0.5, linecolor="red", fmt=".1f")
#6 -> sns.boxplot(x="gender", y="age", data=policeKillingsUS, hue="manner_of_death", palette="PRGn")
#7 -> sns.swarmplot(x="gender", y="age", data=policeKillingsUS, hue="manner_of_death")
#8 -> sns.pairplot(combine_data)
#9 -> sns.countplot(data=policeKillingsUS, x="armed", order=policeKillingsUS.armed.value_counts().iloc[:6].index)
#10 -> sns.barplot(x=x[:20], y=y[:20]) -> when compared to countplot for drawing same graph it is 20x times faster than countplot


#Visualization

#.loc is label-based and .iloc is index-based

# %%


# %%
