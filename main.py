from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

@app.get("/", response_class=HTMLResponse, tags= ['Home'])
async def home():
    """ 
    Pagina de inicio que muestra una presentacion al Proyecto Steam_Games_API_Project
    Returns:
    HTMLResponse: Respuesta HTML que muestra la presentación
    """
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Steam Games API Project</title>
        <style>
            body {
                background-color: #060200;
                color: #FFFFFF;
                font-family: "Segoe UI", sans-serif;
                margin: 0;
                padding: 0;
            }

            header {
                background-color: #163E93;
                text-align: center;
                padding: 20px;
            }

            h1 {
                font-size: 36px;
                margin: 0;
            }

            p {
                font-size: 18px;
                line-height: 1.5;
                padding: 20px;
            }

            button {
                background-color: #30A3DA;
                color: #FFFFFF;
                border: none;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background-color: #051C2A;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Steam Games API Project</h1>
        </header>
        <main>
            <p>Bienvenidos a mi Projecto Individual #01 del Henry DataScience Bootcamp! En este proyecto se trabajo con tres datasets extraidos de la Plataforma Steam. El boton debajo los llevara a la pagina con los diferentes endpoints que se crearon tal lo solicitado por el scope del proyecto. Espero que les resulte interesante.</p>
            <button onclick="window.location.href='docs'">View Documentation</button>
        </main>
    </body>
    </html>
    """ 
    return HTMLResponse(content=template)
  

# Cargamos los dataframes que se utilizan para las funciones
df_games = pd.read_parquet('data/RS_games.parquet')

# 3.2.1 **Developer** number of games and percentage of Free content by developer by year
df_developer = pd.read_parquet('data/developers.parquet')

# 3.2.2 **User_data**: how much money has the user spent, what percentage, from the total number of games the user owns
# has the user recommended from the reviews.recommend and how many games he purchased
df_items_users = pd.read_parquet('data/items_users.parquet')

## 3.2.3 **User_for_genres**: this function must return the user with more minutes accumulated for the given genres
# and a list of minutes accumulated per year since the release date
df_user_id_genres = pd.read_parquet('data/user_id_genres.parquet')

# 3.2.4 **Best_developer_year**: this function returns the top 3 developers 
# based on the largest amount of recommendations for the given year (reviews.recommend = True and sentiment = 2)
df_reviews = pd.read_parquet('data/RS_reviews.parquet')

# 3.2.5 **Developer_Reviews_Analyis**: given a developer, the function returns a 
# dicctionary with developer as keys and the amount of Positive and Negative Sentiment Reviews.
df_best_developers = pd.read_parquet('data/best_developer.parquet')

@app.get(path="/developer",
         description= ''' <font color= '#1E1E24'>
                            Instructions:<br>
                            1. Click on "Try it out" to input Developer Name <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Try any of this developers as example: <br>
                            List [ "Dovetail Games", "Valve", "Kotoshiro", "Capcom", "Ubisoft - San Francisco", "KOEI TECMO GAMES CO., LTD.", 'Stainless Games','DL Softworks','Choice of Games']                     
                            ''',
                            tags=['Consulta Generales'])                                
# 3.2.1 **Developer** number of games and percentage of Free content by developer by year
def developer(developer_name: str = Query(...,
                                          description='developer',
                                          example= 'Dovetail Games')):
    try:
        developer_df = df_developer[df_developer['developer'] == developer_name]

        if developer_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for developer: {developer_name}")

        year_counts = developer_df.groupby('release_year')['item_id'].count().reset_index()
        year_counts.columns = ['Year', 'Games Quantity']

        total_items = year_counts['Games Quantity'].sum()

        free_items = developer_df[developer_df['price'] == 0].groupby('release_year')['item_id'].count().reset_index()
        free_items.columns = ['Year', 'Free Items']

        summary_df = pd.merge(year_counts, free_items, on='Year', how='left')

        summary_df['Free Content (%)'] = (summary_df['Free Items'] / summary_df['Games Quantity']) * 100

        summary_df['Free Items'] = summary_df['Free Items'].fillna(0)
        summary_df['Free Content (%)'] = summary_df['Free Content (%)'].fillna(0)

        # Convert DataFrame to dictionary
        summary_dict = summary_df.to_dict(orient='records')

        return JSONResponse(content=summary_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get(path="/user_data",
         description= ''' <font color= '#1E1E24'>
                            Esta función, al ingresar el id de un usuario nos devuelve <br>
                            - cuanto dinero gastó el usuario en su perfil<br>
                            - El porcentaje de reviews sobre el total de juegos del usuario<br>
                            - La cantidad total de juegos que el usuario posee<br> 
                            **Instructions**:<br>
                            1. Click on "Try it out" to input a user_id <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Here are some examples for you to try: <br>
                            Possible user_id: ['phrostb','thugnificent', 'chidvd', 'piepai' ,'mayshowganmore', 'DeEggMeister', '76561198070585472',
                            'Steamified', 'rawrvixen', 'djnemonic']                      
                            ''',
                            tags=['Consulta Generales'])   
# 3.2.2 **User_data**: how much money has the user spent, what percentage, from the total number of games the user owns
# has the user recommended from the reviews.recommend and how many games he purchased
def user_data(user_id: str = Query(...,
                                          description='user_id',
                                          example= 'piepai')):
        # Filter user items by user_id
    df_user_id_items = df_items_users[df_items_users['user_id'] == user_id]

    # Merge with df_games to add price
    df_user_id_items = df_user_id_items.merge(df_games[['item_id', 'price']], on='item_id', how='left')
    df_user_id_items['price'] = df_user_id_items['price'].fillna(0)
    
    # Merge with df_reviews to add recommend
    df_user_id_items = df_user_id_items.merge(df_reviews[['user_id', 'item_id', 'recommend']], on=['user_id', 'item_id'], how='left')
    df_user_id_items['recommend'] = df_user_id_items['recommend'].fillna(0)
    
    # Calculate total money spent
    money_spent = df_user_id_items['price'].sum()

    # Calculate recommend percentage
    recommend_percentage = round((df_user_id_items['recommend'].sum() / len(df_user_id_items)), 4) * 100

    # Calculate game quantity
    game_quantity = len(df_user_id_items)

    result = {
        'User_Id': f'{user_id}',
        'Money Spent': f'{money_spent:.2f} USD',
        'Recommend Percentage': f'{recommend_percentage:.2f}%',
        'Game Quantity': game_quantity}
    
    return JSONResponse(content=result)


@app.get(path="/userForGenre",
         description= ''' <font color= '#1E1E24'>
                            Esta función, al ingresar el nombre de un género de juegos devuelve, para ese género, el id del usuario con mayor cantidad de horas de juego acumuladas para ese género<br> 
                            **Instructions**:<br>
                            1. Click on "Try it out" to input a genre <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Try any of this genres as example: <br>
                            Possible genres: 'action', 'adventure', 'animation modeling',
       'audio production', 'casual', 'design illustration', 'early access',
       'education', 'free to play', 'indie', 'massively multiplayer',
       'photo editing', 'racing', 'rpg', 'simulation', 'software training',
       'sports', 'strategy', 'utilities', 'video production', 'web publishing' <br>
       Type genre name in lower case<br>                    
                            ''',
                            tags=['Consulta Generales'])   
## 3.2.3 **User_for_genres**: this function must return the user with more minutes accumulated for the given genres
# and a list of minutes accumulated per year since the release date
def userForGenre(genre: str = Query(...,
                                          description='genre',
                                          example= 'action')):
    try:
        user_id_genres = df_user_id_genres
        user_id_genres = user_id_genres.merge(df_games[['item_id', genre, 'release_year']], on='item_id', how='left')
        user_id_genres.dropna(inplace=True)
        user_id_genres = user_id_genres[user_id_genres[genre] == 1]
        grouped_year = user_id_genres.groupby(['user_id', 'release_year'])['playtime_forever'].sum().reset_index()
        grouped_sum = user_id_genres.groupby(['user_id'])['playtime_forever'].sum().reset_index()

        max_user = grouped_sum.loc[grouped_sum['playtime_forever'].idxmax()]
        max_user_id = max_user['user_id']
        max_user_playtime = grouped_year[grouped_year['user_id'] == max_user_id].set_index('release_year')[
                                'playtime_forever'] / 60  # Convert minutes to hours

        max_user_playtime = max_user_playtime.round().astype(int)

        # Create a list of dictionaries containing the playtime per year
        playtime_per_year = [{str(year): hours} for year, hours in max_user_playtime.items()]

        result = {
            f'User_id with more hours played per {genre}': max_user_id,
            'Hours played': playtime_per_year
        }
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
# 3.2.4 **Best_developer_year**: this function returns the top 3 developers 
# based on the largest amount of recommendations for the given year (reviews.recommend = True and sentiment = 2)
@app.get(path="/top_developers",
         description= ''' <font color= '#1E1E24'>
                            Esta función, al ingresar el año, devuelve una lista con los 3 desarrolladores que ese año recibieron mayor cantidad de reseñas positivas<br>
                            **Instructions**:<br>
                            1. Click on "Try it out" to input a valid year <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Years range from : <br>
                            List [2010,2011,2012,2013,2014,2015,2024]                     
                            ''',
                            tags=['Consulta Generales'])  
def top_developers(year: int = Query(...,
                                          description='Year',
                                          example= 2011)):
    try:
        df_developers = df_best_developers
    
        df_developers = df_developers[df_developers['year']== year]
        df_developers = df_developers.dropna()
        grouped = df_developers[(df_developers['sentiment'] ==2) | ((df_developers['recommend'] == True))].groupby('developer').size().reset_index(name='count')

        sorted_developers = grouped.sort_values(by='count', ascending=False)

        top_3_developers = sorted_developers.head(3)

        result = {}
        positions = ['1', '2', '3']

        for i, row in enumerate(top_3_developers['developer']):
            result[f' Top #{positions[i]} Developer '] = row
        if result == {}:
            mensaje_error = 'No information for the requested year'
            return mensaje_error
        else:
            return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get(path="/developer_reviews_analysis",
         description= ''' <font color= '#1E1E24'>
                            Esta función, al ingresar el nombre de un desarrollador devuelve, para ese desarrollador, cuantas reseñas positivas y negativas recibió<br> 
                            **Instructions**:<br>
                            1. Click on "Try it out" to input Developer Name <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Try any of this developers as example: <br>
                            List [ "Dovetail Games", "Valve", "Kotoshiro", "Capcom", "Ubisoft - San Francisco", "KOEI TECMO GAMES CO., LTD.", 'Stainless Games','DL Softworks','Choice of Games']                     
                            ''',
                            tags=['Consulta Generales'])     
# 3.2.5 **Developer_Reviews_Analyis**: given a developer, the function returns a 
# dicctionary with developer as keys and the amount of Positive and Negative Sentiment Reviews.
def developer_reviews_analysis(developer_name: str = Query(...,
                                          description='developer',
                                          example= 'Dovetail Games')):
    try:
        # Filter the DataFrame for the specified developer
        developer_df = df_best_developers[df_best_developers['developer'] == developer_name]

        # Check if the developer_df is empty (i.e., no data for the specified developer)
        if developer_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for developer: {developer_name}")

        # Count occurrences of positive and negative sentiments
        positive_count = int(developer_df[developer_df['sentiment'] == 2]['sentiment'].count())
        negative_count = int(developer_df[developer_df['sentiment'] == 0]['sentiment'].count())

        # Create the dictionary with the counts
        result = {
            developer_name: {
                'Negative': negative_count,
                'Positive': positive_count
            }
        }
        # Return the result as a JSON response
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    
# Recommender System Functions

import re
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(df_games['clean_title'])
from sklearn.metrics.pairwise import cosine_similarity


@app.get(path="/recommender_system_by_game_id",
         description= ''' <font color= '#1E1E24'>
                            Este es nuestro modelo de Recommendación, al ingresar el nombre de un juego devuelve una lista de juegos recomendada similares al juego ingresado<br> 
                            **Instructions**:<br>
                            1. Click on "Try it out" to input game name <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Here is a list of some games you can try: <br>
                            List [ 'Counter-Strike', 'Counter-Strike: Global Offensive', 'Transistor', 'Killing Floor', 'Team Fortress 2','Unturned',
                            'Rust', 'Left 4 Dead 2 ', 'Terraria', 'DayZ','Warframe', 'Borderlands 2', 'The Walking Dead', 'Loadout',
                            'Starbound', 'Grand Theft Auto V']
                                                    ''',
                            tags=['Recommender System'])     
# input is the title we are searching and we will 
def find_similar_games(game_name: str = Query(...,
                                          description='Game name',
                                          example= 'Transistor')):
    title = clean_title(game_name)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:] # list the first five similar titles
    results = df_games.iloc[indices][::-1]
    
    item_id = results.iloc[0]['item_id']
    similar_users = df_reviews[(df_reviews['item_id'] == item_id) & (df_reviews['rating']>=5)]['user_id'].unique()
    similar_user_recs = df_reviews[(df_reviews['user_id'].isin(similar_users)) & (df_reviews['rating'] > 4)]['item_id']

    # Threshold - No lo usamos porque nuestra base de datos tiene muy poca información, sino este codigo refinaria aun mas el filtro
    #similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    #similar_user_recs = similar_user_recs[similar_user_recs > .05]

    all_users = df_reviews[(df_reviews['item_id'].isin(similar_user_recs.index)) & (df_reviews['rating']>4)]
    all_users_recs = all_users['item_id'].value_counts() / len(all_users['user_id'].unique())

    rec_percentages = pd.concat([similar_user_recs,all_users_recs], axis=1)
    rec_percentages.columns = ['similar','all']

    rec_percentages['score'] = rec_percentages['similar']/rec_percentages['all']

    rec_percentages = rec_percentages.sort_values('score', ascending= False)
    top_results = rec_percentages.head(6).merge(df_games, left_index=True, right_on="item_id")[['item_id', 'score','item_name','genres']]
    # Convert the top results to a list of dictionaries
    top_results_list = top_results.to_dict(orient='records')
    if top_results_list == []:
        error_message = 'Game needs more reviews to produce a Recommendation. Please try another.'
        return error_message
    else:
        return JSONResponse(content=top_results_list)


@app.get(path="/recommender_system_by_user_id",
         description= ''' <font color= '#1E1E24'>
                            Este es nuestro modelo de Recommendación, al ingresar el nombre usuarionos devuelve una lista de 5 juegos recomendados segun juegos que le gustaron a usuarios con similares gustos<br> 
                            **Instructions**:<br>
                            1. Click on "Try it out" to input a user_id <br>
                            2. Scroll down to 'Responses' to view the df result<br>
                            3. Here is a list of user_id you can try: <br>
                            ['lachlantatton','Astrelt', 'danthettt', 'Foxxy34', 'originaldog', 'TzakShrike', 'nanakao', '76561198061759775', 'Capscain','SuperSneakCreamPuff','franklinhi','Sakifx9']
                                                    ''',
                            tags=['Recommender System'])     
def find_user_recommendations(user_id: str = Query(...,
                                          description='User_id',
                                          example= 'danthettt')):
    # Get games rated highly by the user
    user_highly_rated_games = df_reviews[(df_reviews['user_id'] == user_id) & (df_reviews['rating'] >= 5)]['item_id'].unique()
    
    # Find similar users who rated these games highly
    similar_users = df_reviews[(df_reviews['item_id'].isin(user_highly_rated_games)) & (df_reviews['rating'] >= 5)]['user_id'].unique()
    
    # Get recommendations from similar users
    similar_user_recs = df_reviews[(df_reviews['user_id'].isin(similar_users)) & (df_reviews['rating'] > 4)]['item_id']
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .01] # threshold that makes the search similarity perfom better
    
    # Get recommendations from all users
    all_users = df_reviews[(df_reviews['item_id'].isin(similar_user_recs.index)) & (df_reviews['rating'] > 4)]
    all_users_recs = all_users['item_id'].value_counts() / len(all_users['user_id'].unique())
    
    # Calculate scores and merge with game information
    rec_percentages = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_percentages.columns = ['similar', 'all']
    rec_percentages['score'] = rec_percentages['similar'] / rec_percentages['all']
    rec_percentages = rec_percentages.sort_values('score', ascending=False)
    
    # Merge with game information and return top recommendations
    top_results = rec_percentages.head(5).merge(df_games, left_index=True, right_on="item_id")[['item_id', 'score', 'item_name', 'genres']]

    # Convert the top results to a list of dictionaries
    top_results_list = top_results.to_dict(orient='records')
    if top_results_list == []:
        error_message = 'User_id Not found. Please try another.'
        return error_message
    else:
        return JSONResponse(content=top_results_list)