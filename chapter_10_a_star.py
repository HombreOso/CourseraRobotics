import heapq

# Documentation of python heapq library: https://docs.python.org/3/library/heapq.html
# heapq.heappush(heap, item) - Push the value item onto the heap, maintaining the heap invariant.
# heapq.heappop(heap) - Pop and return the smallest item from the heap, maintaining the heap invariant.
# heapq.heapify(x) - Transform list x into a heap, in-place, in linear time.
# heapq.heapreplace(heap, item) - Pop and return the smallest item from the heap, and push the new item; the heap size doesn't change.
# heapq.heappushpop(heap, item) - Push item on the heap, then pop and return the smallest item from the heap; the heap size doesn't change.
# heapq.merge(*iterables, key=None, reverse=False) - Merge multiple sorted inputs into a single sorted output (for example, merge timestamped entries from multiple log files). Returns an iterator over the sorted values.
# heapq.nlargest(n, iterable, key=None) - Return a list with the n largest elements from the dataset defined by iterable. key, if provided, specifies a function of one argument that is used to extract a comparison key from each element in iterable (for example, key=str.lower). Equivalent to: sorted(iterable, key=key, reverse=True)[:n].
# heapq.nsmallest(n, iterable, key=None) - Return a list with the n smallest elements from the dataset defined by iterable. key, if provided, specifies a function of one argument that is used to extract a comparison key from each element in iterable (for example, key=str.lower). Equivalent to: sorted(iterable, key=key)[:n].

import math
import os


def read_nodes(filepath):
    """
    Parse a nodes.csv file and return a dictionary of graph nodes.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the nodes.csv file.  Each non-comment
        line must have the form  ID,x,y,heuristic-cost-to-go  where:
          ID                  -- unique integer identifier (1 through N)
          x, y                -- Cartesian coordinates of the node in the plane
          heuristic-cost-to-go -- admissible (never over-estimating) estimate
                                  of the shortest remaining distance to the
                                  goal node (e.g. Euclidean distance); used as
                                  the h(n) term in A*

    Returns
    -------
    dict
        Mapping  {node_id (int): {'x': float, 'y': float, 'h': float}}
    """
    nodes = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            node_id = int(parts[0])
            nodes[node_id] = {
                'x': float(parts[1]),
                'y': float(parts[2]),
                'h': float(parts[3]),
            }
    return nodes


def read_edges(filepath):
    """
    Parse an edges.csv file and return an undirected adjacency list.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the edges.csv file.  Each non-comment
        line must have the form  ID1,ID2,cost  where:
          ID1, ID2 -- integer IDs of the two nodes connected by the edge
          cost     -- non-negative traversal cost for the edge (same in both
                      directions, i.e. the graph is treated as undirected)

    Returns
    -------
    dict
        Mapping  {node_id (int): [(neighbor_id (int), cost (float)), ...]}
        Both directions of each edge are stored so the graph can be traversed
        from either endpoint.
    """
    adj = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            id1, id2, cost = int(parts[0]), int(parts[1]), float(parts[2])
            adj.setdefault(id1, []).append((id2, cost))
            adj.setdefault(id2, []).append((id1, cost))
    return adj


def read_obstacles(filepath):
    """
    Parse an obstacles.csv file and return a list of circular obstacles.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the obstacles.csv file.  Each non-comment
        line must have the form  x,y,diameter  where:
          x, y     -- Cartesian coordinates of the obstacle's centre
          diameter -- full width of the circular obstacle; halved internally
                      to obtain the radius used in collision checks

    Returns
    -------
    list of tuple
        Each element is  (cx (float), cy (float), radius (float))  where
        radius = diameter / 2.
    """
    obstacles = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            cx, cy, diameter = float(parts[0]), float(parts[1]), float(parts[2])
            obstacles.append((cx, cy, diameter / 2.0))
    return obstacles


def segment_intersects_circle(p1, p2, cx, cy, radius):
    """
    Test whether a line segment intersects a circle.

    Parameters
    ----------
    p1 : tuple of float  (x, y)
        Start point of the line segment.
    p2 : tuple of float  (x, y)
        End point of the line segment.
    cx : float
        x-coordinate of the circle's centre.
    cy : float
        y-coordinate of the circle's centre.
    radius : float
        Radius of the circle (must be > 0).

    Returns
    -------
    bool
        True if any point on the finite segment p1->p2 lies on or inside the
        circle; False if the segment misses the circle entirely.
    """
    # Parametrise the segment as  Q(t) = p1 + t*(p2 - p1),  t in [0, 1].
    # d = p2 - p1  is the direction vector of the segment.
    # f = p1 - C   is the vector from the circle centre C to the segment start.
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - cx
    fy = p1[1] - cy

    # Intersection requires  |Q(t) - C|^2 = r^2, i.e.
    #   |f + t*d|^2 = r^2
    #   (d·d) t^2 + 2(f·d) t + (f·f - r^2) = 0
    # Identify the quadratic coefficients a, b, c:
    a = dx * dx + dy * dy          # |d|^2  (always > 0 for a non-degenerate segment)
    b = 2.0 * (fx * dx + fy * dy)  # 2 (f · d)
    c = fx * fx + fy * fy - radius * radius  # |f|^2 - r^2

    # The discriminant  D = b^2 - 4ac  determines the number of real roots:
    #   D < 0  =>  no real intersection (segment misses the circle entirely)
    #   D = 0  =>  tangent (one touch point)
    #   D > 0  =>  two intersection points at parameters t1 <= t2
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    # Roots via the quadratic formula  t = (-b ± sqrt(D)) / (2a)
    t1 = (-b - sqrt_disc) / (2.0 * a)   # entry parameter (smaller)
    t2 = (-b + sqrt_disc) / (2.0 * a)   # exit  parameter (larger)

    # The segment intersects the circle when the interval [t1, t2] (the chord
    # through the circle) overlaps [0, 1] (the finite segment).
    # Three distinct cases cover all overlapping configurations:
    #   • t1 in [0,1]: the entry point lies on the segment
    #   • t2 in [0,1]: the exit  point lies on the segment
    #   • t1 < 0 and t2 > 1: segment is fully inside the circle
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)


def is_collision_free(n1, n2, nodes, obstacles):
    """
    Check whether the straight-line edge between two graph nodes is obstacle-free.

    Parameters
    ----------
    n1 : int
        ID of the first node (edge start).
    n2 : int
        ID of the second node (edge end).
    nodes : dict
        Node dictionary as returned by read_nodes(); used to look up the (x, y)
        coordinates of n1 and n2.
    obstacles : list of tuple
        Obstacle list as returned by read_obstacles(), each element being
        (cx, cy, radius).

    Returns
    -------
    bool
        True if the segment from n1 to n2 does not intersect any obstacle
        circle; False if at least one obstacle blocks the edge.
    """
    p1 = (nodes[n1]['x'], nodes[n1]['y'])
    p2 = (nodes[n2]['x'], nodes[n2]['y'])
    return not any(
        segment_intersects_circle(p1, p2, cx, cy, r)
        for cx, cy, r in obstacles
    )


def reconstruct_path(parent, goal_id):
    """
    Trace parent pointers from the goal back to the start and return the path.

    Parameters
    ----------
    parent : dict
        Mapping  {node_id (int): predecessor_id (int) or None}  as built
        during the A* search.  The start node maps to None, which terminates
        the back-trace.
    goal_id : int
        ID of the goal node from which the back-trace begins.

    Returns
    -------
    list of int
        Ordered list of node IDs from start to goal (inclusive).
    """
    path = []
    node = goal_id
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def a_star(nodes, adj, obstacles, start_id, goal_id, max_count=10_000):
    """
    Graph-based A* search following the provided pseudocode.

    OPEN is a min-heap keyed by  f(n) = past_cost(n) + h(n), where h(n) is
    the admissible heuristic stored in the node dictionary.  A visited set
    plays the role of "previously occupied C-space grid cells" from the
    pseudocode, preventing a node from being expanded more than once.

    Parameters
    ----------
    nodes : dict
        Node dictionary as returned by read_nodes().  Keys are integer node
        IDs; values are dicts with fields 'x', 'y', and 'h' (heuristic).
    adj : dict
        Adjacency list as returned by read_edges().  Keys are integer node
        IDs; values are lists of (neighbor_id, edge_cost) tuples.
    obstacles : list of tuple
        Obstacle list as returned by read_obstacles(), each element being
        (cx, cy, radius).  Edges that intersect an obstacle are skipped.
    start_id : int
        ID of the node from which the search begins (past_cost = 0).
    goal_id : int
        ID of the node the search is trying to reach.  The search terminates
        as soon as this node is popped from OPEN.
    max_count : int, optional
        Maximum number of nodes that may be expanded before declaring failure.
        Mirrors the MAXCOUNT guard in the pseudocode.  Default is 10 000.

    Returns
    -------
    list of int or None
        Ordered list of node IDs from start_id to goal_id if a path is found,
        or None if the search exhausts OPEN or reaches max_count without
        finding the goal.
    """
    # Step 1-3: initialise
    open_heap = []                          # (f, node_id)
    heapq.heappush(open_heap, (nodes[start_id]['h'], start_id))
    past_cost = {start_id: 0.0}
    parent    = {start_id: None}
    visited   = set()
    counter   = 1

    # Step 4: main loop
    while open_heap and counter < max_count:
        _, current = heapq.heappop(open_heap)   # step 5

        # Step 6-8: goal check
        if current == goal_id:
            return reconstruct_path(parent, goal_id)

        # Step 9: skip if already expanded
        if current in visited:
            continue

        # Step 10-11: mark as visited
        visited.add(current)
        counter += 1

        # Step 12-19: expand neighbours (each edge is a "control")
        for neighbor, edge_cost in adj.get(current, []):
            if neighbor in visited:
                continue

            # Step 14: collision check
            if not is_collision_free(current, neighbor, nodes, obstacles):
                continue

            # Step 15: cost of the path to q_new
            new_cost = past_cost[current] + edge_cost

            # Only update if we found a cheaper route
            if neighbor not in past_cost or new_cost < past_cost[neighbor]:
                past_cost[neighbor] = new_cost
                parent[neighbor]    = current
                f_new = new_cost + nodes[neighbor]['h']
                heapq.heappush(open_heap, (f_new, neighbor))   # step 16-17

    return None   # step 22: FAILURE


def write_path(filepath, path, start_id):
    """
    Write the solution path to a path.csv file.

    Parameters
    ----------
    filepath : str
        Absolute or relative path of the output file to write (created or
        overwritten).
    path : list of int or None
        Ordered sequence of node IDs representing the solution path as
        returned by a_star().  If None (no solution was found), only
        start_id is written so the visualiser can still show the start node.
    start_id : int
        ID of the start node, used as the sole entry when path is None.

    Returns
    -------
    None
    """
    with open(filepath, 'w') as f:
        if path:
            f.write(','.join(str(n) for n in path) + '\n')
        else:
            # No solution: write start node only
            f.write(str(start_id) + '\n')


def path_cost(path, adj):
    """
    Compute the total cost of a path by summing its edge weights.

    Parameters
    ----------
    path : list of int
        Ordered sequence of node IDs as returned by a_star() or
        reconstruct_path().  Must contain at least two nodes.
    adj : dict
        Adjacency list as returned by read_edges(), used to look up the cost
        of each consecutive pair of nodes in the path.

    Returns
    -------
    float
        Sum of the edge costs for all consecutive node pairs in path.
    """
    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        cost = next(c for nb, c in adj[u] if nb == v)
        total += cost
    return total


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_dir  = os.path.join(script_dir, 'Chapter_10_A-Star', 'Scene5_example')

    nodes     = read_nodes    (os.path.join(scene_dir, 'nodes.csv'))
    adj       = read_edges    (os.path.join(scene_dir, 'edges.csv'))
    obstacles = read_obstacles(os.path.join(scene_dir, 'obstacles.csv'))

    # Start = node 1 (highest heuristic / bottom-left corner)
    # Goal  = node with heuristic 0 (top-right corner)
    start_id = 1
    goal_id  = min(nodes, key=lambda n: nodes[n]['h'])

    print(f"Start : node {start_id}  ({nodes[start_id]['x']}, {nodes[start_id]['y']})")
    print(f"Goal  : node {goal_id}  ({nodes[goal_id]['x']}, {nodes[goal_id]['y']})")
    print(f"Obstacles: {len(obstacles)}")

    path = a_star(nodes, adj, obstacles, start_id, goal_id)

    path_file = os.path.join(scene_dir, 'path.csv')
    write_path(path_file, path, start_id)

    if path:
        cost = path_cost(path, adj)
        print(f"\nSUCCESS")
        print(f"Path  : {' -> '.join(str(n) for n in path)}")
        print(f"Cost  : {cost:.4f}")
        print(f"Nodes visited: {len(path)}")
    else:
        print("\nFAILURE: no path found.")

    print(f"\npath.csv written to:\n  {path_file}")


if __name__ == '__main__':
    main()
