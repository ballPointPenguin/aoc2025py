class UnionFind:
    def __init__(self, n):
        """Initialize with n elements (0 to n-1)."""
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        """Find the root / representative of element x with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union two components. Returns True if they were separate, False if connected"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # already in same component

        # Union by rank (keep tree balanced)
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        return True  # unioned

    def get_component_sizes(self):
        """Return list of component sizes."""
        root_sizes = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            root_sizes[root] = self.size[root]
        return list(root_sizes.values())
