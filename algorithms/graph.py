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
