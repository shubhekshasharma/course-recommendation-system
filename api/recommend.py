import sys
import os
import json
from http.server import BaseHTTPRequestHandler

# Resolve repo root and set CWD so relative pickle/CSV paths in recommend.py work
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

from openai import OpenAI
from recommendations.recommend import (
    find_courses_by_same_cluster,
    find_courses_by_text_similarity,
    find_courses_by_preferred_credit_level,
)


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["LLM_LITE_TOKEN"].strip(),
        base_url=os.environ["LLM_LITE_URL"].strip(),
    )


def _get_workload_and_keywords(client: OpenAI, user_input: str) -> dict:
    prompt = f"""
    You are a course recommendation system for a university. You have access to a database of courses with their descriptions and credit hours.
    For this task, you will only determine the workload user prefers based on their input.

    Note when determining the workload, we have 3 categories:
    - Low
    - Standard
    - High

    Do not assume other workload categories exist.
    The user's input is as follows:
    {user_input}

    Your tasks:
    1. Determine the workload preference (must be one of: Low, Standard, High)
    2. Extract and infer academic interest keywords for course matching

    WORKLOAD CATEGORIES:
    - Low: Light workload, fewer courses, more flexibility
    - Standard: Moderate workload, typical course load
    - High: Heavy workload, many courses, intensive study

    IMPORTANT: If the user does not mention anything about workload, effort level, time commitment, or course load, default to "Standard".

    INTEREST KEYWORDS EXTRACTION:
    Extract comprehensive academic keywords that represent the user's interests. These keywords will be used to match against course descriptions and titles.

    Guidelines for keyword extraction:
    - Include ALL relevant subject areas, topics, and fields mentioned or implied
    - Add related academic disciplines and subdisciplines
    - Include synonyms and related terminology commonly used in course catalogs
    - Focus ONLY on academic/subject matter keywords (exclude workload preferences, time commitments, or vague terms)
    - Use specific academic terms that would appear in course titles and descriptions
    - Include technical terminology, techniques, and methodologies specific to the subject
    - Add variations and related forms of words (e.g., "photograph", "photography", "photographic", "photographer")
    - Think about what words would appear in a dedicated course focused entirely on this subject
    - Prioritize depth over breadth - include 15-20 highly relevant keywords rather than generic terms

    Examples:
    Input: "I'm interested in the Oscars and film studies"
    Keywords: ["film", "cinema", "movies", "acting", "theatre", "performing arts", "drama", "entertainment", "media studies", "visual arts", "filmmaking", "cinematography", "directing", "screenwriting", "film production", "film theory", "film history"]

    Input: "Interstellar travel and space exploration"
    Keywords: ["space", "astronomy", "astrophysics", "aerospace", "rockets", "spacecraft", "cosmology", "planetary science", "physics", "engineering", "exploration", "interstellar", "orbital", "satellites", "propulsion"]

    Input: "Machine learning and AI for healthcare applications"
    Keywords: ["machine learning", "artificial intelligence", "AI", "healthcare", "medical", "health", "data science", "computer science", "biomedical", "algorithms", "neural networks", "deep learning", "clinical", "medicine", "predictive modeling", "diagnostics"]

    Input: "I like photography"
    Keywords: ["photography", "photographic", "camera", "digital photography", "portrait", "documentary photography", "photojournalism", "image making", "visual storytelling", "photo editing", "composition", "lighting", "darkroom", "photographer", "photographs", "images", "picture taking", "photo critique"]

    Return the result in JSON format:
    {{
        "category": "Low" | "Standard" | "High",
        "reasoning": "Explanation of why this workload category was chosen based on the user's input.",
        "interest_key_words": ["film", "acting", "theatre", "performing arts"]
    }}
    """

    response = client.chat.completions.create(
        model="GPT 4.1 Mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that recommends courses for a university."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=300,
    )
    return json.loads(response.choices[0].message.content)


def _get_reasoning(client: OpenAI, results, user_input: str) -> list:
    records = results.to_dict(orient="records")

    prompt = f"""
    You are a course recommendation system for a university.
    Our ML models recommended the following courses based on the user's input:
    {records}

    Now, provide a brief reasoning for why each course was recommended based on the user's input below:
    {user_input}

    Return the result in JSON format as a list of objects with the following fields:
    {{
        "key": "Course ID",
        "title": "Course Title",
        "description": "Course Description",
        "minimum_credits": "Min credits",
        "similarity": similarity score in float with 2 decimal places,
        "reasoning": "Explanation of why this course was recommended based on the user's input."
    }}

    Make a limit of 4 recommendations only in your result.
    If you notice a long course description, summarize it.

    Ensure it is a valid JSON string.

    Note: if there are no input records then do not hallucinate. In that case you should return no results.
    """

    response = client.chat.completions.create(
        model="GPT 4.1 Mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that recommends courses for a university."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=10000,
    )
    return json.loads(response.choices[0].message.content)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))
            user_input = body.get("user_input", "").strip()

            if not user_input:
                return self._respond(400, {"error": "user_input is required."})

            client = _get_client()

            response_dict = _get_workload_and_keywords(client, user_input)
            credit_category = response_dict["category"]
            keywords_text = " ".join(response_dict["interest_key_words"])

            clustered = find_courses_by_same_cluster(keywords_text)
            similar = find_courses_by_text_similarity(
                user_text=keywords_text,
                same_cluster_courses=clustered,
                top_k=25,
            )
            filtered = find_courses_by_preferred_credit_level(
                similar_courses=similar,
                preferred_credit_level=credit_category,
            )

            recommendations = _get_reasoning(client, filtered, keywords_text)

            self._respond(200, {
                "recommendations": recommendations[:4],
                "similar_courses": similar.to_dict(orient="records"),
                "preferred_credit_level": credit_category,
            })

        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
