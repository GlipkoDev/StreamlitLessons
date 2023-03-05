import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA players stats explorer')

st.markdown("""
This app performs web-scraping of NBA players stats data!
* **Used libraries**:  pandas, streamlit, numpy, matplotlib, seaborn
* **Data source**: [Basketball-reference.com](https://www.basketball-reference.com/)
""")

st.sidebar.header('User input features')
years = range(2023,1989,-1)
year = st.sidebar.selectbox('Year', years)

@st.cache
def scrape_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url)
    raw = html[0]
    raw = raw.drop(raw[raw['Age'] == 'Age'].index)
    raw = raw.fillna(0)
    return raw.drop(columns='Rk')

df = scrape_data(year)
team_names = sorted(df['Tm'].unique())
teams = st.sidebar.multiselect('Teams', team_names, team_names)


position_names = ['C', 'SG', 'PF', 'PG', 'SF']
positions = st.sidebar.multiselect('Positions', position_names, position_names)

team_filter = df['Tm'].isin(teams)
position_regex = "|".join(positions)
position_filter = df['Pos'].str.contains(position_regex, regex=True)

df = df[team_filter & position_filter]

st.write(f'Filtered data dimension: {df.shape[0]} rows and {df.shape[1]} columns')
st.write(df.head())

st.download_button('Download filtered data', df.to_csv(), 'Filtered_data.csv')

if st.button('Show correlation map between numeric data'):
    df.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)
