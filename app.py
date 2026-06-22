from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# ── Load & prepare data ──────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "Movies.csv")

df = pd.read_csv(CSV_PATH)
movies = df[["title", "overview", "release_date", "vote_average",
             "vote_count", "popularity"]].dropna(subset=["title", "overview"])
movies = movies.reset_index(drop=True)

tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["overview"])
cosine_sim = cosine_similarity(tfidf_matrix)

movie_indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# ── Helper ───────────────────────────────────────────────────────────────────

def recommend_movies(movie_title: str, n: int = 10):
    movie_title = movie_title.strip()

    if movie_title in movie_indices:
        idx = movie_indices[movie_title]
    else:
        matches = movies[movies["title"].str.lower().str.contains(
            movie_title.lower(), na=False)]
        if matches.empty:
            return None, []
        idx = matches.index[0]
        movie_title = matches.iloc[0]["title"]

    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n + 1]
    rec_indices = [i[0] for i in sim_scores]
    scores = [round(float(s[1]), 4) for s in sim_scores]

    recs = movies.iloc[rec_indices].copy()
    recs["similarity_score"] = scores

    def safe_year(d):
        try:
            return str(pd.to_datetime(d).year)
        except Exception:
            return "N/A"

    results = []
    for _, row in recs.iterrows():
        results.append({
            "title": row["title"],
            "similarity_score": row["similarity_score"],
            "year": safe_year(row.get("release_date")),
            "vote_average": round(float(row["vote_average"]), 1) if pd.notna(row.get("vote_average")) else None,
            "overview": row["overview"][:200] + "…" if len(str(row["overview"])) > 200 else row["overview"],
        })

    return movie_title, results


def get_suggestions(query: str, limit: int = 8):
    q = query.strip().lower()
    mask = movies["title"].str.lower().str.startswith(q)
    hits = movies[mask]["title"].tolist()[:limit]
    if len(hits) < limit:
        extra = movies[movies["title"].str.lower().str.contains(q, na=False)]["title"].tolist()
        for h in extra:
            if h not in hits:
                hits.append(h)
            if len(hits) >= limit:
                break
    return hits[:limit]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True)
    movie_title = data.get("movie", "").strip()
    n = int(data.get("n", 10))

    if not movie_title:
        return jsonify({"error": "No movie title provided."}), 400

    matched_title, results = recommend_movies(movie_title, n)

    if matched_title is None:
        return jsonify({"error": f"'{movie_title}' not found in the dataset."}), 404

    return jsonify({"matched_title": matched_title, "recommendations": results})


@app.route("/suggest", methods=["GET"])
def suggest():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    return jsonify(get_suggestions(q))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)