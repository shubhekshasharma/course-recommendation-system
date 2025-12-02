import pickle
import pandas as pd
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import scipy


COURSES_FILES_PATH = "/Users/subu/Desktop/Subu/Duke/DESIGNTK_530/designtk-530-f1/courses_with_cluster.csv"

df = pd.read_csv(COURSES_FILES_PATH)

kmeans = pickle.load(open("pickles/k_means_course_clusters.pkl", "rb"))

vectorizer = pickle.load(open("pickles/tfidf_vectorizer.pkl", "rb"))

course_vectors = pickle.load(open("pickles/course_vectors.pkl", "rb"))

course_credits_model = pickle.load(open("pickles/random_forest_course_credits.pkl", "rb"))


def find_courses_by_same_cluster(user_text):
    vec = vectorizer.transform([user_text])
    user_cluster = kmeans.predict(vec)[0]
    print("User input belongs to cluster:", user_cluster)
    same_cluster_courses = df[df["cluster"] == user_cluster]
    return same_cluster_courses


def find_courses_by_text_similarity(user_text, same_cluster_courses, top_k=10):
    # Convert user text to TF-IDF
    user_vec = vectorizer.transform([user_text])

    # Filling with empty string
    same_cluster_courses["description"] = same_cluster_courses["description"].fillna("")

    # Compute TF-IDF vectors for ONLY same-cluster subset
    cluster_vectors = vectorizer.transform(same_cluster_courses["description"])

    # Compute similarity
    sims = cosine_similarity(user_vec, cluster_vectors).flatten()

    # Top K ranked indices
    top_idx = sims.argsort()[::-1][:top_k]

    # Build result based on the 
    results = same_cluster_courses.iloc[top_idx][["key", "title", "description", "minimum credits"]].copy()
    results["similarity"] = sims[top_idx]

    return results



def find_courses_by_preferred_credit_level(similar_courses, preferred_credit_level: str = None):
    # Use same TF-IDF vectorizer on course descriptions
    desc_vecs = vectorizer.transform(similar_courses["description"])

    X = scipy.sparse.hstack([desc_vecs])

    similar_courses["predicted_credit_level"] = course_credits_model.predict(X)

    if preferred_credit_level:
        return similar_courses.loc[
            similar_courses["predicted_credit_level"] == preferred_credit_level
        ]

    return similar_courses
