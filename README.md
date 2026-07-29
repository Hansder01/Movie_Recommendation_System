# 🎬 CineMatch: Semantic Movie & TV Show Recommendation System

Welcome to **CineMatch**, an advanced semantic recommendation engine for Movies and TV Shows. 
This application utilizes modern Natural Language Processing (NLP) to understand the descriptions, genres, cast, and metadata of media to serve highly accurate and context-aware recommendations.

Coupled with a visually stunning **Cosmic Glassmorphism UI**, the system provides a seamless and futuristic user experience for discovering your next binge-watch.

---

## ✨ Features

- **Semantic Search & Embeddings**: Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to generate dense vector embeddings of movie descriptions and metadata, capturing the true contextual meaning of the content.
- **Fast Similarity Matching**: Employs scikit-learn's `NearestNeighbors` (K-NN) algorithm to instantly calculate cosine similarity distances between thousands of embeddings.
- **Dynamic Scoring Engine**: Augments the base vector similarity score with smart bonuses for exact genre matches, shared directors, intersecting cast members, and chronological proximity (release year).
- **Automated Caching**: Embeddings and the K-NN model are cached to disk (`models/`) on the first run, dramatically speeding up subsequent application launches.
- **Modern UI**: A completely bespoke "Cosmic Glass" frontend aesthetic utilizing dynamic background elements, glassmorphism blur effects, custom hover micro-interactions, and neon gradient text.
- **Clickable Rabbit Holes**: Every recommendation card acts as a new search query, allowing infinite exploration.

---

## 🛠️ Tech Stack

- **Backend / Routing**: Python 3, Flask
- **Machine Learning & NLP**: SentenceTransformers, Scikit-learn, PyTorch
- **Data Manipulation**: Pandas, Numpy
- **Frontend**: Vanilla HTML5, CSS3, Jinja2 Templating

---

## 📂 Project Structure

```text
Movie-Recommendation-system/
├── app.py                     # Flask application entry point and routing
├── model.py                   # Core Recommender logic and dynamic scoring
├── embedding_generator.py     # SentenceTransformer embedding & caching logic
├── feature_builder.py         # Data preprocessing and feature engineering
├── config.py                  # Global configurations and paths
├── requirements.txt           # Project dependencies
├── data/
│   └── titles.csv             # The dataset (Movies and TV Shows)
├── models/                    # Cached model files (.pkl, .npy) (Generated on run)
├── static/
│   └── style.css              # Custom Cosmic Glass CSS theme
└── templates/
    ├── base.html              # Global Jinja layout (Navbar, Footer, Background)
    ├── index.html             # The main search interface
    └── results.html           # The recommendations display interface
```

---

## 🚀 Installation & Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd Movie-Recommendation-system
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   python app.py
   ```
   *Note: The very first time you run the application, it may take a minute or two to download the Transformer model, build the feature strings, and generate the embeddings. These are saved to the `models/` directory so all future launches will be nearly instantaneous.*

5. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000/` in your web browser.

---

## 💡 Usage

1. Type the name of a Movie or TV Show into the search bar.
2. Click **Recommend**.
3. View the source title details at the top of the results page.
4. Explore the **"Because you liked..."** section below to find highly similar content. 
5. Click on any recommendation card to instantly pivot your search and find recommendations for *that* specific title.

---

## 📝 Notes & Troubleshooting

- **Movie Not Found?**: If a title is searched that does not exist within the dataset, you will gracefully be notified that the title has not been adapted yet. Try searching for popular Netflix originals!
- **UI Not Updating?**: If you make CSS changes, remember to hard-refresh your browser (`Ctrl+F5` or `Cmd+Shift+R`) to clear the cached stylesheet.

---

## 🤝 Contributing
Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

## 👨‍💻 Author
**Sayan Bhanja Chowdhury**

MCA Student | Python Developer | Machine Learning Enthusiast

---

## ⭐ Support
If you found this project helpful, consider giving it a ⭐ on GitHub.
