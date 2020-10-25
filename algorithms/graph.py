from collections import defaultdict

class UndirectedGraph:
    def __init__(self, vertices):
        self.vertices = set(vertices)
        self.V = len(self.vertices)
        self.graph = defaultdict(set) # default dictionary to store graph

    def addEdge(self, u, v):
        self.graph[u].add(v)
        self.graph[v].add(u)

    def BFSUtil(self, v, visited, nodes):
        # Initialization
        queue = []
        visited.add(v)
        queue.append(v)

        # BFS
        while len(queue) > 0:
            s = queue.pop(0)
            nodes.add(s)

            for i in self.graph[s]:
                if not i in visited:
                    visited.add(i)
                    queue.append(i)

    def getSCCs(self):
        visited = set()

        # Now process all vertices in order defined by Stack
        sccs = []
        for v in self.vertices:
            if not v in visited:
                scc = set()
                self.BFSUtil(v, visited, scc)
                sccs.append(scc)

        return sccs

class DirectedGraph:
    def __init__(self, vertices):
        self.vertices = set(vertices)
        self.V = len(self.vertices)
        self.graph = defaultdict(set) # default dictionary to store graph

    def addEdge(self, u, v):
        self.graph[u].add(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited.add(v)

        # Recur for all the vertices adjacent to this vertex
        for u in self.graph[v]:
            if not u in visited:
                self.topologicalSortUtil(u, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0,v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = set()
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for v in self.vertices:
            if not v in visited:
                self.topologicalSortUtil(v, visited, stack)

        # Print contents of stack
        return stack
