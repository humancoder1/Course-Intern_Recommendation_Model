import pandas as pd
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List





dataframe1 = pd.read_csv("./all_courses.csv")


new_df = dataframe1.drop(columns=["Rating" ,"Review Count","Prerequisites" , "Affiliates" , "Type" , "Description" , "Duration" , "URL"]).copy()
## Analysis Conclusion 

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(new_df["Skills Covered"])
cosine_sim = cosine_similarity(matrix , matrix)

def get_course(skills):
    selected_skills = ', '.join(skills)

    user_vector = vectorizer.transform([selected_skills])
    # print("User Vector shape:", user_vector.shape)
    # print("Matrix shape:", matrix.shape)
    cosine_sim_with_selected_skills = cosine_similarity(user_vector, matrix)

    sim_scores = list(enumerate(cosine_sim_with_selected_skills[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    course_indices = [score[0] for score in sim_scores[:5]]
    recommendations = new_df['Title'].iloc[course_indices].tolist()

    return recommendations


# print(get_course(["CSS"]))

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommendation_func/{skills}")
def recommendation_func(skills : str):
    recommendations = get_course([skills])
    return recommendations
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
