import pickle
import time
import heapq
from collections import deque

start_word = "roughly"
goal_word = "punishment"
algorithm = "dfs"
DEPTH_LIMIT = 20

print("Loading neighbor graph")
with open("data/neighbors.pkl", "rb") as f:
    neighbors = pickle.load(f)

def bfs(start, goal, neighbors):
    queue = deque([start])
    visited = set([start])
    parent = {}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(parent, start, goal), nodes_expanded

        for n in neighbors.get(current, []):
            if n not in visited:
                visited.add(n)
                parent[n] = current
                queue.append(n)

    return None, nodes_expanded

def dfs(start, goal, neighbors):
    stack = [(start, 0)]
    visited = set()
    parent = {}
    nodes_expanded = 0

    while stack:
        current, depth = stack.pop()

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(parent, start, goal), nodes_expanded

        if depth < DEPTH_LIMIT:
            for n in neighbors.get(current, []):
                if n not in visited:
                    parent[n] = current
                    stack.append((n, depth + 1))

    return None, nodes_expanded

def ucs(start, goal, neighbors):
    pq = [(0, start)]
    parent = {}
    cost = {start: 0}
    nodes_expanded = 0

    while pq:
        g, current = heapq.heappop(pq)
        nodes_expanded += 1

        if current == goal:
            return reconstruct_path(parent, start, goal), nodes_expanded

        for n in neighbors.get(current, []):
            new_cost = g + 1
            if n not in cost or new_cost < cost[n]:
                cost[n] = new_cost
                parent[n] = current
                heapq.heappush(pq, (new_cost, n))

    return None, nodes_expanded

def reconstruct_path(parent, start, goal):
    path = []
    current = goal

    while current != start:
        path.append(current)
        if current not in parent:
            return None
        current = parent[current]

    path.append(start)
    path.reverse()
    return path

algorithms = {
    "BFS": bfs,
    "DFS": dfs,
    "UCS": ucs
}

for name, func in algorithms.items():

    print("Running", name)

    start_time = time.time()
    path, expanded = func(start_word, goal_word, neighbors)
    end_time = time.time()

    if path:
        print("Path found:")
        print(" -> ".join(path))
        print("Length:", len(path))
    else:
        print("No path found")

    print("Nodes expanded:", expanded)
    print("Runtime:", end_time - start_time)
    print("\n")