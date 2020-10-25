# %%

import pandas as pd #data processing, csv file I/O 
import numpy as np #linear algebra
import matplotlib.pyplot as plt
import plotly.graph_objs as go #import graph object as "go"
from plotly.offline import plot, iplot
# %%

#read data from .csv
timesData = pd.read_csv("./data/timesData.csv")
earthquakes = pd.read_csv("./data/database.csv")

# %%

#------->Line plot

#prepare data frame
df = timesData.iloc[:100]

#layout declarations
tick = {
        "ticks" : "",
        "tickcolor" : "crimson",
        "tickwidth" : 2,
        "ticklen" : 4
        }

trace1 = go.Scatter(
                    x=df.world_rank,
                    y=df.citations,
                    mode="lines",
                    marker=dict(color="#3345df"),
                    name="citations",
                    text=df.university_name)
trace2 = go.Scatter(
                    x=df.world_rank,
                    y=df.teaching,      
                    mode="lines",
                    marker=dict(color="#e32f12"),
                    name="teaching",
                    text=df.university_name)
data = [trace1, trace2]

layout = dict(
            title="Citation and Teaching vs World Rank of top 100 Universities",
            xaxis={**tick, "title" : "World Rank"},
            yaxis={**tick, "title" : "Value"})

fig = dict(data=data, layout=layout)
iplot(fig)


# %%

#------->Scatter plot

#better way 
# filtered = timesData[(timesData["year"] < 2017) & (timesData["year"] > 2013)]

# fig = go.Figure()

# for year, df in filtered.groupby("year"):
#         fig.add_scatter(
#                 x=df.iloc[:100].world_rank,
#                 y=df.iloc[:100].citations,
#                 name=year,
#                 mode="markers"
#         )

#normal way
df_2014 = timesData[timesData["year"] == 2014].iloc[:100]
df_2015 = timesData[timesData["year"] == 2015].iloc[:100]
df_2016 = timesData[timesData["year"] == 2016].iloc[:100]


trace0 = go.Scatter(
                x=df_2014.world_rank,
                y=df_2014.citations,
                name="2014",
                mode="markers",
                marker=dict(size=5),
                text=df_2014.university_name

)

trace1 = go.Scatter(
                x=df_2015.world_rank,
                y=df_2015.citations,
                name="2015",
                mode="markers",
                marker=dict(size=7),
                text=df_2015.university_name

)

trace2 = go.Scatter(
                x=df_2016.world_rank,
                y=df_2016.citations,
                name="2016",
                mode="markers",
                marker=dict(size=9),
                text=df_2016.university_name
)

data = [trace0, trace1, trace2]

layout = dict(
        title="title about something",
        xaxis=dict(title="rank"),
        yaxis=dict(title="citations"))

fig = dict(data=data, layout=layout)

iplot(fig)

# %%

#------->barplot

top_5 = timesData.iloc[:5]
top_5.head(10)

#normal way

trace0 = go.Bar(
        x=top_5.university_name,
        y=top_5.teaching,
        name="teaching",
        text=top_5.country
)

trace1 = go.Bar(
        x=top_5.university_name,
        y=top_5.citations,
        name="citations",
        text=top_5.country
)

#dictionaries way
# temp = timesData[timesData["year"] == 2014].iloc[:3]

# data_dict = {
#       0 : {
#         "y" : temp.research,
#         "x" : temp.university_name,
#         "name" : "research",
#         "type" : "bar"
#       },

#       1 : {
#         "y" : temp.total_score,
#         "x" : temp.university_name,
#         "name" : "total_score",
#         "type" : "bar"
#       }     
# }
# data = [data_dict[0], data_dict[1]]
# layout = {
#         "title" : "dictionaries way relative bar plot",
#         "barmode" : "group",
#         "xaxis" : {
#                 "tickangle" : 45
#         }
# }
# fig = go.Figure(data=data, layout=layout)
# iplot(fig)


#loop way
# fig = go.Figure()

# param = ["citations", "teaching"]

# for itr in range(2):
#         fig.add_bar(
#                 x=top_5.university_name,
#                 y=top_5[param[itr]]
#         )
#visualization


data = [trace0, trace1]
layout = dict(
        barmode="relative",
        title="bar plot example",
        )        
fig = dict(data=data, layout=layout)

iplot(fig)

# %%

#-------> Subplots example 1

from plotly.subplots import make_subplots

df = timesData[timesData["year"] == 2015].iloc[:7]

#trace declarations
trace0 = go.Scatter(
                x=df.research,
                y=df.university_name,
                text=[each for each in df.research],
                textposition="top center",
                mode="lines+markers+text",
                name="research"
)

trace1 = go.Bar(
                x=df.income,
                y=df.university_name,
                text=[each for each in df.income],
                textposition="outside",
                name="income",
                orientation="h",
                
)

#visualization
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

#add traces
fig.add_trace(trace1, 1, 1)
fig.add_trace(trace0, 1, 2)

#update x axis properties
fig.update_xaxes(title_text= "research", range=[0, 120], row=1, col=1)
fig.update_xaxes(title_text= "income", row=1, col=2, side="top")

#update y axis properties
fig.update_yaxes(title_text="university name", row=1, col=1, domain=(0, .78))
fig.update_yaxes(showline=True, linewidth=2, linecolor="gray", domain=(0, .78))


fig.update_layout(
                title="Subplots example",
                width = 800, 
                height=600, 
                paper_bgcolor="white", 
                plot_bgcolor="white"
)

iplot(fig)


# %%

#-------> Subplots example 2

x = [1, 2, 3, 4, 5, 6]
y = [11, 22, 34, 44, 50, 60]


fig = make_subplots(rows=2, cols=3, specs=[
                                        [{"colspan" : 2, "rowspan" : 2}, None, {}],
                                        [None, None, {}]
                                        ], subplot_titles=("first", "second", "third"))

trace0 = go.Scatter(
        x=x,
        y=y,
        name="it is x"
)

trace1 = go.Scatter(
        x=x,
        y=y,
        name="it is x"
)

trace2 = go.Scatter(
        x=x,
        y=y,
        name="it is x"
)

fig.add_trace(trace0, 1, 1)
fig.add_trace(trace1, 1, 3)
fig.add_trace(trace2, 2, 3)



fig.show()

# %%

#------->Pie chart example
# pie_trace = go.Pie(labels=["ozzy", "morr", "iss"], values=[2, 1, 1])
# bar_trace = go.Bar(x=[1, 2, 3], y=[5, 6, 7], name="something")

# fig = make_subplots(rows=1, cols=2, specs=[[{"type" : "domain"}, {}]])

# fig.add_trace(pie_trace, row=1, col=1)
# fig.add_trace(bar_trace, row=1, col=2)

# fig.update_traces(hoverinfo="label+percent", hole=.3, pull=[0, 0, 0.1], textinfo="percent", marker=dict(line=dict(color="gray", width=1.5)), row=1, col=1)
# fig.update_yaxes(domain=(0, .85), row=1, col=2)

# fig.show()

df = timesData[timesData["year"] == 2016].iloc[:7]

pie_values = [each.replace(",", ".") for each in df.num_students]
pie_labels = df.university_name.values

pie_trace = go.Pie(labels=pie_labels, values=pie_values)

#visualization

fig = go.Figure()

fig.add_trace(pie_trace)

fig.update_traces(hole=.3, hoverinfo="label+percent")


fig.update_layout(
        width=800,
        title_text="Pie chart example", 
        annotations=[dict(text="n of s", x=0.5, y=0.5, font_size=20, showarrow=False)],
        legend=dict(x=1, y=1.2))

fig.show()

# %%

#------->Bubble Chart example

#prepare data frame

df = timesData[timesData["year"] == 2016].iloc[:20]

bubble_trace = go.Scatter(
        x=df.world_rank,
        y=df.teaching,
        mode="markers",
        marker=dict(
                size=[float(each.replace(",", ".")) for each in df.num_students],
                color=[float(each) for each in df.international],
                showscale=True)       
)

#visualization

fig = go.Figure()

fig.add_trace(bubble_trace)

fig.update_layout(title_text="world rank vs teaching with number of students")

fig.show()

# %%

#-------> Histogram example

#prepare data
x2011 = timesData.student_staff_ratio[timesData["year"] == 2011]
x2012 = timesData.student_staff_ratio[timesData["year"] == 2012]

#create traces
trace0 = go.Histogram(
        x=x2011,
        name="2011",
        opacity=0.7
)

trace1 = go.Histogram(
        x=x2012,
        name="2012",
        opacity=0.9
)

#visualization
fig = go.Figure()
fig.add_trace(trace0)
fig.add_trace(trace1)
fig.update_layout(
        barmode="overlay", 
        xaxis=dict(title="student staff ratio"), 
        yaxis=dict(title="count"), 
        title="Student staff ratio histogram 2011 and 2012")
fig.show()

#%%

#------->World Cloud example

from wordcloud import WordCloud

#prepare data
wordc = timesData.country[timesData["year"] == 2011]

plt.figure(figsize=(10,10))
x = ["asd", "asda"]
wordcloud = WordCloud(width=800, height=800, background_color="white").generate(" ".join(wordc))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# %%

#------->Box plot example

#prepare data
df = timesData[timesData["year"] == 2015]
y0 = df.total_score
y1 = df.research

#traces
trace0 = go.Box(y=y0, name="total_score")
trace1 = go.Box(y=y1, name="research")

#visualization
fig = go.Figure()
fig.add_traces(trace0)
fig.add_traces(trace1)
fig.show()

# %%
#------->Scatter matrix plots

from plotly import figure_factory as ff

#prepare data
df = timesData.loc[timesData["year"] == 2015, ["research", "international", "total_score"]].reset_index(drop=True)
df["index"] = np.arange(1, len(df) + 1)

#visualization
fig = ff.create_scatterplotmatrix(df, diag="box", index="index", width=700, height=700, colormap="Portland", colormap_type="cat")
fig.show()


# %%

#------->Inset plots

#prepare data
df = timesData[timesData["year"] == 2015].iloc[:200]

trace0 = go.Scatter(
        x=df.world_rank,
        y=df.teaching,
        name="teaching",
        mode="lines"
)

trace1 = go.Scatter(
        x=df.world_rank,
        y=df.income,
        name="income",
        mode="lines",
        xaxis="x2",
        yaxis="y2",
        text=df.university_name
)

fig = go.Figure()
fig.add_traces([trace0, trace1])
fig.update_layout(
        xaxis2=dict(
                domain=[.6, .95],
                anchor="y2" #xaxis2: {anchor: 'y2'} means that xaxis2 joins yaxis2 at the origin.
        ),
        yaxis2=dict(
                domain=[.6, .95],
                anchor="x2"
        ),
        title="Some title",
        width=1000)
fig.show()

# %%

#------->Scatter3D Plots

#prepare data
df = timesData[timesData["year"] == 2015].iloc[:100]
df.iloc[39].world_rank = 39
df.world_rank = df.world_rank.astype(int)

trace0 = go.Scatter3d(
        x=df.world_rank,
        y=df.teaching,
        z=df.international,
        mode="markers",
        marker=dict(
                size=3.3,
                color=df.research,
                colorscale="Viridis",
                showscale=True
        ),
)

fig = go.Figure()
fig.add_trace(trace0)
fig.update_layout(      
        title="3d scatter plot",
        margin=dict(
                t=50,
                b=0,
                l=0,
                r=0
        )
)

fig.show()

# %%
#------->Multiple Subplots in same frame

#prepare data
df = timesData[timesData["year"] == 2014].iloc[:100]

trace0 = go.Scatter(
        x=df.world_rank,
        y=df.research,
        name="research"
)

trace1 = go.Scatter(
        x=df.world_rank,
        y=df.total_score,
        name="total score",
        xaxis="x2",
        yaxis="y2"
)
trace2 = go.Scatter(
        x=df.world_rank,
        y=df.citations,
        name="citations",
        xaxis="x3",
        yaxis="y3"
)
trace3 = go.Scatter(
        x=df.world_rank,
        y=df.teaching,
        name="teaching",
        xaxis="x4",
        yaxis="y4"
)

fig = go.Figure()
fig.add_traces([trace0, trace1, trace2, trace3])
fig.update_layout(
        xaxis=dict(                     #sol alt
                domain=[0, .45]
        ),
        yaxis=dict(
                domain=[0, .45]
        ),
        xaxis2=dict(                    #sağ alt
                domain=[.55, 1],
                anchor="y2"
        ),
        yaxis2=dict(
                domain=[0, .45],
                anchor="x2"
        ),
        xaxis3=dict(                    #sağ üst
                domain=[.55, 1],
                anchor="y3"
        ),
        yaxis3=dict(
                domain=[.55, 1],
                anchor="x3"
        ),
        xaxis4=dict(                    #sol üst
                domain=[0, .45],
                anchor="y4"
        ),
        yaxis4=dict(
                domain=[.55, 1],
                anchor="x4"
        ),
        title="Multiple subplots in same frame"
)

fig.show()


# %%

#------->Map plot example 1 

#read .csv file
aerial = pd.read_csv("./data/operations.csv")
weather = pd.read_csv("./data/Summary of Weather.csv")
weather_station_locations = pd.read_csv("./data/Weather Station Locations.csv")
weather_station_locations.head()

#data cleaning
keep_list = ["Mission Date", "Theater of Operations", "Country", "Air Force", 
        "Aircraft Series", "Callsign", "Takeoff Base", "Takeoff Location",
        "Takeoff Latitude", "Takeoff Longitude", "Target Country", "Target City",
        "Target Type", "Target Industry", "Target Priority", "Target Latitude", "Target Longitude"]
aerial = aerial.loc[:, keep_list] #keep columns which in list
aerial.dropna(subset=["Country", "Target Longitude", "Takeoff Longitude"], inplace=True) #drop rows that has nan values in subset columns
aerial = aerial[aerial.iloc[:, 8].isin(["4248"]) == False] #drop 4248 value from takeoff latitude
aerial = aerial[aerial.iloc[:, 9].isin([1355]) == False] #drop 1355 value from takeoff longitude

weather_station_locations = weather_station_locations.loc[:, ["WBAN", "NAME", "STATE/COUNTRY ID", "Latitude", "Longitude"]]

weather = weather.loc[:, ["STA", "Date", "MeanTemp"]]
#cleaning completed
#set color for countries
aerial["color"] = ""
aerial.color[aerial["Country"] == "USA"] = "red"
aerial.color[aerial["Country"] == "GREAT BRITAIN"] = "yellow"
aerial.color[aerial["Country"] == "NEW ZEALAND"] = "purple"
aerial.color[aerial["Country"] == "SOUTH AFRICA"] = "green"

trace0 = go.Scattergeo(
        lon=aerial["Takeoff Longitude"],
        lat=aerial["Takeoff Latitude"],
        hoverinfo="text",
        text="Country: " + aerial.Country + " - Takeoff Location: " + aerial["Takeoff Location"] + " - Takeoff Base: " + aerial['Takeoff Base'],
        mode="markers",
        marker=dict(
                sizemode="area",
                sizeref=1,
                size=10,
                line=dict(
                        width=1,
                        color="#ccc"
                        ),
                opacity=0.75,
                color=np.where(aerial.Country == "USA", "red", "green")
        )
)

fig = go.Figure()

fig.add_trace(trace0)

fig.update_layout(
        title="Aerial attacks takeoff bases in WW2",
        width=800,
        hovermode="closest",
        geo=dict(
                # projection=dict(type=""),
                showland=True, landcolor="LightGreen",       
                showocean=True, oceancolor="LightBlue",
                showcountries=True,
                resolution=110
                # showlakes=True, lakecolor="blue",
                # showrivers=True, rivercolor="blue",
                # showcoastlines=True, coastlinecolor="purple",
                # showsubunits=True, subunitwidth=1 #subunits of country
                
        ),
        margin=dict(
                t=55,
                b=0,
                r=0,
                l=0
        )
              
)

fig.show()


# %%

#------->Map plot example 2 

#prepare data
aerial = aerial.iloc[:100]

trace0 = go.Scattergeo(
        lat=aerial["Takeoff Latitude"],
        lon=aerial["Takeoff Longitude"],
        hoverinfo="text",
        hovertext="Attacking Country: " + aerial.Country + " Takeoff Location: " + aerial["Takeoff Location"] + " Takeoff Base: " + aerial["Takeoff Base"]  ,
        mode="markers",
        marker=dict(
                color="blue",
                size=5
        ),
)
trace1 = go.Scattergeo(
        lat=aerial["Target Latitude"],
        lon=aerial["Target Longitude"],
        hoverinfo="text",
        hovertext="Target Country: " + aerial["Target Country"] + " Target City: " + aerial["Target City"],
        mode="markers",
        marker=dict(
                color="red",
                size=1
        ),
)

fig = go.Figure()

for i, row in aerial.iterrows():
        fig.add_trace(
                go.Scattergeo(
                        lon=[row["Takeoff Longitude"], row["Target Longitude"]],
                        lat=[row["Takeoff Latitude"], row["Target Latitude"]],
                        mode="lines",
                        line=dict(
                                color="black",
                                width=.7,
                        ),
                        opacity=.7                        
                )
        )
fig.add_traces([trace0, trace1])
fig.update_layout(
        geo=dict(
                showland=True, landcolor="#e0dbb1",
                # projection=dict(
                #         type=""
                # ),
                resolution=50
        ),
        showlegend=False,
        hovermode="closest"     
)

fig.show()

# %%

#------->Animations Example
#we should use plot method(not iplot) to show animated graphs 

fig = go.Figure(
    data=[go.Scatter(x=[0, 1], y=[0, 1])],
    layout=go.Layout(
        xaxis=dict(range=[0, 5], autorange=False),
        yaxis=dict(range=[0, 5], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None])])]
    ),
    frames=[go.Frame(data=[go.Scatter(x=[1, 2], y=[1, 2])]),
            go.Frame(data=[go.Scatter(x=[1, 4], y=[1, 4])]),
            go.Frame(data=[go.Scatter(x=[3, 4], y=[3, 4])])]
)

plot(fig)

# %%

#-------> Homework




 # %%
