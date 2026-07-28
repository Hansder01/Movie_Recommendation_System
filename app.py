from flask import Flask, render_template, request, jsonify

from model import recommend, load

app = Flask(__name__)

# Load model once when Flask starts
load()

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        movie = request.form["movie"]

        recommendations = recommend(movie)

    return render_template(
        "index.html",
        recommendations=recommendations
    )

@app.route("/api/recommend")
def api():

    title = request.args.get("title")

    return jsonify(recommend(title))

if __name__ == "__main__":
    app.run(debug=True)