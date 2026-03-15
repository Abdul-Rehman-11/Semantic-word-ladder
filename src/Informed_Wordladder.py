import numpy as np
import pickle
import heapq
import time

def load_glove(file_name):

    word_list = []
    vector_list = []

    print("Loading GloVe file...")

    with open(file_name, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            word_list.append(parts[0])
            vector_list.append(np.array(parts[1:], dtype=float))

    vector_list = np.array(vector_list)

    vector_list = vector_list / np.linalg.norm(vector_list, axis=1, keepdims=True)

    word_index = {}
    for i in range(len(word_list)):
        word_index[word_list[i]] = i

    return word_list, vector_list, word_index

def cosine(v1, v2):
    return np.dot(v1, v2)

# Heuristic Function
# h(n) = 1 - cosine(n, goal)
def get_heuristic(word, goal, word_index, vectors):

    v1 = vectors[word_index[word]]
    v2 = vectors[word_index[goal]]

    return 1 - cosine(v1, v2)

# Greedy Best First Search
def greedy(start, goal, neighbors, word_index, vectors):

    if start not in word_index or goal not in word_index:
        print("Start or goal word not in vocabulary.")
        return None, 0

    pq = []
    heapq.heappush(pq, (0, start))

    visited = set()
    parent = {}
    expanded = 0

    while pq:

        h_value, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path, expanded

        for next_word in neighbors[current]:

            if next_word not in visited:
                h = get_heuristic(next_word, goal, word_index, vectors)
                parent[next_word] = current
                heapq.heappush(pq, (h, next_word))

    return None, expanded

# A* Search
def astar(start, goal, neighbors, word_index, vectors):

    if start not in word_index or goal not in word_index:
        print("Start or goal word not in vocabulary.")
        return None, 0

    pq = []
    heapq.heappush(pq, (0, start))

    parent = {}
    g_cost = {}
    g_cost[start] = 0

    visited = set()
    expanded = 0

    while pq:

        f_value, current = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path, expanded

        for next_word in neighbors[current]:

            sim = cosine(
                vectors[word_index[current]],
                vectors[word_index[next_word]]
            )

            step_cost = 1 - sim
            new_cost = g_cost[current] + step_cost

            if next_word not in g_cost or new_cost < g_cost[next_word]:

                g_cost[next_word] = new_cost
                parent[next_word] = current

                h = get_heuristic(next_word, goal, word_index, vectors)
                f = new_cost + h

                heapq.heappush(pq, (f, next_word))

    return None, expanded

if __name__ == "__main__":

    start_word = "pjl"
    goal_word = "lemmen"

    with open("neighbors.pkl", "rb") as f:
        neighbors = pickle.load(f)

    words, vectors, word_index = load_glove("glove.100d.20000.txt")

    print("\nGreedy")
    t1 = time.time()
    path, nodes = greedy(start_word, goal_word, neighbors, word_index, vectors)
    t2 = time.time()

    print("Path:", path)
    print("Length:", len(path) if path else 0)
    print("Nodes Expanded:", nodes)
    print("Time:", t2 - t1)

    print("\nA*...")
    t1 = time.time()
    path, nodes = astar(start_word, goal_word, neighbors, word_index, vectors)
    t2 = time.time()

    print("Path:", path)
    print("Length:", len(path) if path else 0)
    print("Nodes Expanded:", nodes)
    print("Time:", t2 - t1)