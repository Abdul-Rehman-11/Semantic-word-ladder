import streamlit as st
import pickle
import time

from Informed_Wordladder import greedy, astar
from Uninformed_Wordladder import bfs, dfs, ucs  
from Informed_Wordladder import load_glove 

@st.cache_resource
def load_data():

    with open("data/neighbors.pkl", "rb") as f:
        neighbors = pickle.load(f)

    words, vectors, word_index = load_glove("data/glove.100d.20000.txt")

    return neighbors, words, vectors, word_index


neighbors, words, vectors, word_index = load_data()

st.title("Word Ladder Search in Embedding Space")

start_word = st.text_input("Enter Start Word:")
goal_word = st.text_input("Enter Goal Word:")

algorithm = st.selectbox(
    "Select Search Algorithm",
    ["BFS", "DFS", "Greedy", "A*", "UCS"]
)

run_button = st.button("Run Search")

if run_button:

    if start_word not in word_index or goal_word not in word_index:
        st.error("Start or Goal word not in vocabulary.")
    else:

        start_time = time.time()

        if algorithm == "BFS":
            path, expanded = bfs(start_word, goal_word, neighbors)

        elif algorithm == "DFS":
            path, expanded = dfs(start_word, goal_word, neighbors)

        elif algorithm == "UCS":
            path, expanded = ucs(start_word, goal_word, neighbors)

        elif algorithm == "Greedy":
            path, expanded = greedy(start_word, goal_word, neighbors, word_index, vectors)

        elif algorithm == "A*":
            path, expanded = astar(start_word, goal_word, neighbors, word_index, vectors)

        end_time = time.time()


        if path:
            st.success("Path Found!")
            st.write("Word Ladder:", " → ".join(path))
            st.write("Length:", len(path))
        else:
            st.warning("No Path Found.")

        st.write("Nodes Expanded:", expanded)
        st.write("Time Taken:", end_time - start_time)