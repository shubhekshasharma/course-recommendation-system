import pickle
import pandas as pd
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import scipy


COURSES_FILES_PATH = "courses_with_cluster.csv"

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
    """
    Find courses by matching user keywords against course titles, descriptions, and codes.

    Uses title/keyword matching (NOT TF-IDF) for better accuracy
    Prioritizes courses with subject keywords in title/code
    Returns similarity scores from 0-100% based on keyword matches

    Returns DataFrame with top_k courses sorted by similarity score
    """
    # Prepare data
    same_cluster_courses = same_cluster_courses.copy()
    same_cluster_courses["description"] = same_cluster_courses["description"].fillna("")
    same_cluster_courses["title"] = same_cluster_courses["title"].fillna("")

    user_keywords = set(user_text.lower().split())
    words_list = user_text.lower().split()

    # Score each word based on, Position, Length
    word_scores = []

    for i, word in enumerate(words_list):
        if len(word) > 5: 
            position_score = max(100 - i, 0)
            length_score = len(word) * 5
            total_score = position_score + length_score
            word_scores.append((word, total_score))

    word_scores.sort(key=lambda x: x[1], reverse=True)
    num_core = max(3, min(8, int(len(word_scores) * 0.4)))
    core_keywords = [word for word, score in word_scores[:num_core]]

    # Fallback: if no core keywords found, use all long words
    if len(core_keywords) == 0:
        core_keywords = [w for w in user_keywords if len(w) > 7]

    final_sims = np.zeros(len(same_cluster_courses))

    for idx, (i, row) in enumerate(same_cluster_courses.iterrows()):
        title_lower = str(row['title']).lower()
        title_words = set(title_lower.split())
        key_lower = str(row['key']).lower()
        desc_lower = str(row['description']).lower()

        # Scoring logic (designed so subject-specific courses get 80-95% similarity)
        # Core keyword in title (70% max
        # If title contains ANY core subject keyword, it's highly relevant
        has_core_in_title = any(kw in title_lower for kw in core_keywords)
        if has_core_in_title:
            # Count how many core keywords appear
            # Base 70% for having core keyword, +extra for multiple matches
            core_matches = sum(1 for kw in core_keywords if kw in title_lower)
            core_title_score = 0.70 + min((core_matches - 1) * 0.05, 0.15)
        else:
            core_title_score = 0

        # Course code/prefix matching (15% max)
        # Courses with subject in code (e.g., "photo") are dedicated to that subject
        code_match_score = 0
        if any(kw in key_lower for kw in core_keywords):
            code_match_score = 0.15

        # General keyword overlap in title (10% max)
        # Additional points for other related keywords in title
        if len(user_keywords) > 0:
            non_core_words = user_keywords - set(core_keywords)
            if len(non_core_words) > 0:
                non_core_overlap = len(non_core_words & title_words) / len(non_core_words)
                title_overlap_score = non_core_overlap * 0.10
            else:
                title_overlap_score = 0
        else:
            title_overlap_score = 0

        # Keyword frequency in description (5% max)
        # Courses that frequently mention keywords in description
        keyword_freq = sum(desc_lower.count(kw) for kw in core_keywords if len(kw) > 6)
        freq_score = min(keyword_freq / 20.0, 1.0) * 0.05

        # Combine all scores (max possible = 100% if all criteria met)
        # A course with subject keyword in BOTH title AND code can score 85-100%
        final_sims[idx] = core_title_score + code_match_score + title_overlap_score + freq_score

    # Top K ranked indices based on final similarity, and build result with final similarity scores
    top_idx = final_sims.argsort()[::-1][:top_k]
    
    results = same_cluster_courses.iloc[top_idx][
        ["key", "title", "description", "minimum credits", "cluster"]
    ].copy()
    results["similarity"] = final_sims[top_idx]  # Use final combined similarity

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
