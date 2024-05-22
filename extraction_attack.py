import torch


def similarity(img1, img2, threshold=20.0):
    # img1 and img2 are tensors of shape (3, N, N)
    # divide each into 16 non-overlapping N // 4 x N // 4 tiles
    tile_size = img1.shape[1] // 4
    if tile_size < 1:
        return False
    tiles1 = img1.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    tiles2 = img2.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    tiles1 = tiles1.permute(1, 2, 0, 3, 4)
    tiles2 = tiles2.permute(1, 2, 0, 3, 4)
    tiles1 = tiles1.reshape(16, -1)
    tiles2 = tiles2.reshape(16, -1)

    # compute l2 distance across all corresponding pairs of tiles
    # if we compared tiles with different locations
    # even the same image would appear to be different
    l2_norms = torch.norm(tiles1 - tiles2, dim=-1)
    max_l2 = l2_norms.max().item()

    if max_l2 < threshold:
        return True
    else:
        return False


def construct_graph(images, threshold=20.0):
    graph = {}
    for i in range(len(images)):
        img1 = images[i]
        graph[i] = []
        for j in range(len(images)):
            if i != j:
                img2 = images[j]
                if similarity(img1, img2, threshold=threshold):
                    graph[i].append(j)
    return graph


def find_cliques(graph):
    cliques = []
    visited = set()

    def dfs(node, clique):
        visited.add(node)
        clique.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, clique)

    for node in graph:
        if node not in visited:
            clique = set()
            dfs(node, clique)
            if len(clique) > 1:
                cliques.append(clique)

    return cliques


def attack(images, min_clique_size=3, threshold=20.0):
    graph = construct_graph(images, threshold=threshold)
    cliques = find_cliques(graph)
    if len(cliques) > 0:
        largest_clique = max(cliques, key=len)
    else:
        largest_clique = set()
    if len(largest_clique) >= min_clique_size:
        # return the index of the first example in the clique
        return list(largest_clique)[0]
    else:
        return -1
