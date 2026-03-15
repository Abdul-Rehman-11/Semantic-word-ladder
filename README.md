# Semantic-word-ladder
Semantic Word Ladder Search with multiple AI Search Algorithms, Interactive UI, and Visualization

**Project Description**

Semantic Word Ladder is a project that finds transformation paths between words based on their meaning rather than just spelling.It combines graph algorithms, AI search strategies, and word embeddings to create a fast, interactive word ladder solver with visualization and performance optimization.

Unlike traditional word ladder puzzles (letter-based), this project uses GloVe embeddings to navigate a graph of semantically similar words, allowing for meaningful transformations.

**Features**
*Implements Uninformed Search Algorithms:*
BFS (Breadth-First Search)
DFS (Depth-First Search)
UCS (Uniform Cost Search)

*Implements Informed (Heuristic) Search Algorithms:*
Greedy Best-First Search
A* Search Algorithm

*Optimized BFS can search 17,000+ nodes in ~0.035s*
*Interactive GUI using Streamlit for testing start/goal words and algorithms*
*Displays search results, including:*
Word Ladder path
Number of nodes expanded
Time taken

**Repository Structure**
semantic-word-ladder/
│
├── src/                     
│   ├── build_graph.py           # Builds the neighbor graph from GloVe embeddings
│   ├── informed_wordladder.py   # Greedy & A* search implementations
│   ├── uninformed_wordladder.py # BFS, DFS, UCS search implementations
│   └── gui.py                   # Streamlit interface for interactive search
│
├── data/                        
│   ├── glove.100d.20000.txt     # GloVe embeddings
│   └── neighbors.pkl             # Precomputed neighbor graph
│
├── .gitignore                   
├── LICENSE                      
└── README.md     

**Installation**
1) Clone the repo:
  git clone https://github.com/Abdul-Rehman-11/Semantic-word-ladder.git
  cd semantic-word-ladder
2) Create a virtual environment (recommended):
  python -m venv venv
  source venv/bin/activate       # Linux/macOS
  venv\Scripts\activate          # Windows
3) Install dependencies:
   numpy
   streamlit

**How to Use**
1) Build Neighbor Graph:
   python src/build_graph.py
this generate neighbors.pkl from glove.100d.20000.txt
2) Run GUI:
   streamlit run src/gui.py
3) Run from terminal (optional):
   python src/informed_wordladder.py
  python src/uninformed_wordladder.py
