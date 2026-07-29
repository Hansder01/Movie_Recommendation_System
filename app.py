from flask import Flask, render_template, request, jsonify

from model import recommend, load

app = Flask(__name__)

# Load model once when Flask starts
load()

@app.route("/", methods=["GET", "POST"])
def home():
    movie = request.form.get("movie") or request.args.get("movie")
    
    if movie:
        source_movie, recommendations = recommend(movie)
        
        return render_template(
            "results.html",
            source_movie=source_movie,
            recommendations=recommendations,
            searched_query=movie
        )

    return render_template("index.html")

@app.route("/api/recommend")
def api():
    title = request.args.get("title")
    source, recs = recommend(title)
    return jsonify({
        "source": source,
        "recommendations": recs
    })

if __name__ == "__main__":
    app.run(debug=True)