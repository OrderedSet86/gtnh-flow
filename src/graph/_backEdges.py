import collections

# Modified from https://stackoverflow.com/a/53995651/7247528


class BasicGraph:
    def __init__(self, edges) -> None:
        self.edges = edges
        self.adj = BasicGraph._build_adjacency_list(edges)
        self.back_edges = []

    @staticmethod
    def _build_adjacency_list(edges) -> dict[str, list[str]]:
        adj = collections.defaultdict(list)
        for edge in edges:
            adj[edge[0]].append(edge[1])
        return adj


def dfs(G: BasicGraph) -> None:
    # Mutates G directly
    discovered = set()
    finished = set()

    for u in list(G.adj):
        if u not in discovered and u not in finished:
            dfs_visit(G, u, discovered, finished)


def dfs_visit(
        G: BasicGraph,
        u: dict[str, list[str]], # adjacency list
        discovered: set[str],
        finished: set[str],
    ) -> None:
    # Mutates G directly
    discovered.add(u)

    for v in G.adj[u]:
        # Detect cycles
        if v in discovered:
            G.back_edges.append((u, v))
            # print(f"Cycle detected: found a back edge from {u} to {v}.")
            continue

        # Recurse into DFS tree
        if v not in finished:
            dfs_visit(G, v, discovered, finished)

    discovered.remove(u)
    finished.add(u)


if __name__ == "__main__":
    G = BasicGraph(
        [
            ('u', 'v'),
            ('u', 'x'),
            ('v', 'y'),
            ('w', 'y'),
            ('w', 'z'),
            ('x', 'v'),
            ('y', 'x'),
            ('z', 'z'),
        ]
    )

    dfs(G)