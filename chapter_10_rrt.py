import math
import os
import random
from typing import Optional


# ── shared geometric helpers (reused from chapter_10_a_star.py) ───────────────


def read_obstacles(filepath: str) -> list[tuple[float, float, float]]:
    """
    Parse an obstacles.csv file and return a list of circular obstacles.

    Parameters
    ----------
    filepath : str
        Path to obstacles.csv.  Each non-comment line must have the form
        x,y,diameter where (x,y) is the centre and diameter is the full width.

    Returns
    -------
    list of tuple
        Each element is (cx, cy, radius) with radius = diameter / 2.
    """
    obstacles: list[tuple[float, float, float]] = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            cx, cy, diameter = float(parts[0]), float(parts[1]), float(parts[2])
            obstacles.append((cx, cy, diameter / 2.0))
    return obstacles


def read_initial_nodes(filepath: str) -> dict[int, dict[str, float]]:
    """
    Parse a minimal nodes.csv file to obtain the start and goal positions.

    The file is expected to have exactly two rows:
      • The start node (node 1) at the beginning of the path.
      • The goal node (the node whose heuristic-cost-to-go equals 0).

    Parameters
    ----------
    filepath : str
        Path to nodes.csv.  Each non-comment line has the form
        ID,x,y,heuristic-cost-to-go.

    Returns
    -------
    dict
        Mapping {node_id (int): {'x': float, 'y': float, 'h': float}}.
    """
    nodes: dict[int, dict[str, float]] = {}
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


def segment_intersects_circle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """
    Test whether a line segment intersects a circle.

    Uses a parametric quadratic formulation: Q(t) = p1 + t*(p2 - p1), t in [0,1].
    The segment intersects the circle when the chord interval [t1, t2] overlaps
    the finite segment interval [0, 1].

    Parameters
    ----------
    p1 : tuple of float  (x, y)
        Start point of the segment.
    p2 : tuple of float  (x, y)
        End point of the segment.
    cx : float
        x-coordinate of the circle centre.
    cy : float
        y-coordinate of the circle centre.
    radius : float
        Radius of the circle (must be > 0).

    Returns
    -------
    bool
        True if any point on p1->p2 lies on or inside the circle.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - cx
    fy = p1[1] - cy

    # Quadratic coefficients of |Q(t) - C|^2 = r^2
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return False

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # Segment intersects circle if either root lies in [0,1] or the segment is
    # fully enclosed (t1 < 0 and t2 > 1).
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0) or (t1 < 0.0 and t2 > 1.0)


# ── RRT-specific helpers ──────────────────────────────────────────────────────


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Compute the Euclidean distance between two 2-D points.

    This distance is used both as the metric for finding the nearest tree node
    and as the edge cost when a new node is added to the tree.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first point.
    x2, y2 : float
        Coordinates of the second point.

    Returns
    -------
    float
        Non-negative Euclidean distance.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def sample_random_point(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    goal_x: float,
    goal_y: float,
    goal_bias: float,
) -> tuple[float, float]:
    """
    Sample a point uniformly at random from the configuration space, with an
    optional bias toward the goal.

    With probability goal_bias the goal position itself is returned, which
    helps pull the tree toward the goal and speeds up convergence.  Otherwise
    a uniformly random point from the rectangular region [x_min,x_max] x
    [y_min,y_max] is returned.

    Parameters
    ----------
    x_min, x_max : float
        Horizontal bounds of the sampling region.
    y_min, y_max : float
        Vertical bounds of the sampling region.
    goal_x, goal_y : float
        Coordinates of the goal point used for biased sampling.
    goal_bias : float
        Probability in [0, 1] of returning the goal position directly.

    Returns
    -------
    tuple of float  (x, y)
        The sampled configuration.
    """
    if random.random() < goal_bias:
        # Biased sample: return the goal position to attract the tree
        return goal_x, goal_y
    # Uniform sample from the search space
    return random.uniform(x_min, x_max), random.uniform(y_min, y_max)


def find_nearest_node(
    tree_nodes: dict[int, dict[str, float]],
    x_samp: float,
    y_samp: float,
) -> int:
    """
    Find the node in the tree that is nearest to a sampled point.

    Performs a linear scan of all tree nodes and returns the ID of the one
    whose Euclidean distance to (x_samp, y_samp) is smallest.

    Parameters
    ----------
    tree_nodes : dict
        Current tree nodes mapping {node_id: {'x': float, 'y': float}}.
    x_samp, y_samp : float
        Coordinates of the sampled configuration x_samp.

    Returns
    -------
    int
        Node ID of the nearest tree node.
    """
    nearest_id = -1
    min_dist = float('inf')
    for node_id, node in tree_nodes.items():
        d = euclidean_distance(node['x'], node['y'], x_samp, y_samp)
        if d < min_dist:
            min_dist = d
            nearest_id = node_id
    return nearest_id


def steer(
    x_near: float,
    y_near: float,
    x_samp: float,
    y_samp: float,
    step_size: float,
) -> tuple[float, float]:
    """
    Compute a new configuration x_new by moving from x_nearest toward x_samp
    by at most step_size (the local planner).

    If x_samp is already within step_size of x_nearest, x_new = x_samp
    (we can reach x_samp in one step).  Otherwise x_new is placed exactly
    step_size away from x_nearest along the straight line to x_samp.

    Parameters
    ----------
    x_near, y_near : float
        Coordinates of the nearest tree node x_nearest.
    x_samp, y_samp : float
        Coordinates of the sampled configuration x_samp.
    step_size : float
        Maximum distance to advance from x_nearest toward x_samp.

    Returns
    -------
    tuple of float  (x_new, y_new)
        Coordinates of the new candidate configuration.
    """
    d = euclidean_distance(x_near, y_near, x_samp, y_samp)
    if d == 0.0:
        # x_samp coincides with x_nearest; return it unchanged
        return x_samp, y_samp
    if d <= step_size:
        # x_samp is reachable in one step
        return x_samp, y_samp
    # Advance step_size along the direction from x_nearest to x_samp
    ratio = step_size / d
    x_new = x_near + ratio * (x_samp - x_near)
    y_new = y_near + ratio * (y_samp - y_near)
    return x_new, y_new


def is_motion_collision_free(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    obstacles: list[tuple[float, float, float]],
) -> bool:
    """
    Check whether the straight-line motion from (x1,y1) to (x2,y2) is
    collision-free with respect to all circular obstacles.

    The check is performed by testing whether the line segment intersects any
    obstacle circle.  A motion that would pass through or end inside an
    obstacle is considered a collision.

    Parameters
    ----------
    x1, y1 : float
        Start coordinates of the motion (x_nearest).
    x2, y2 : float
        End coordinates of the motion (x_new).
    obstacles : list of tuple
        Each element is (cx, cy, radius) as returned by read_obstacles().

    Returns
    -------
    bool
        True if the motion does not intersect any obstacle; False otherwise.
    """
    p1 = (x1, y1)
    p2 = (x2, y2)
    return not any(
        segment_intersects_circle(p1, p2, cx, cy, r)
        for cx, cy, r in obstacles
    )


def is_in_goal_region(
    x: float,
    y: float,
    goal_x: float,
    goal_y: float,
    tolerance: float,
) -> bool:
    """
    Test whether a configuration lies within the goal region X_goal.

    The goal region is defined as a disc of radius tolerance centred on the
    goal position.

    Parameters
    ----------
    x, y : float
        Coordinates of the configuration to test.
    goal_x, goal_y : float
        Coordinates of the goal position.
    tolerance : float
        Radius of the goal region; a configuration is considered to have
        reached the goal when its distance to the goal is <= tolerance.

    Returns
    -------
    bool
        True if (x, y) is inside the goal region.
    """
    return euclidean_distance(x, y, goal_x, goal_y) <= tolerance


# ── core RRT algorithm ────────────────────────────────────────────────────────


def rrt(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    obstacles: list[tuple[float, float, float]],
    step_size: float = 0.05,
    goal_tolerance: float = 0.05,
    max_tree_size: int = 5_000,
    goal_bias: float = 0.1,
    x_min: float = -0.5,
    x_max: float = 0.5,
    y_min: float = -0.5,
    y_max: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[dict[int, dict[str, float]], list[tuple[int, int, float]], Optional[list[int]]]:
    """
    Rapidly-exploring Random Tree (RRT) path planner.

    Builds a tree rooted at the start configuration by iteratively sampling
    random configurations, steering toward them from the nearest tree node,
    and adding collision-free extensions.  The algorithm terminates when a
    node lands inside the goal region X_goal or the tree reaches max_tree_size.

    Algorithm steps follow the pseudocode in RRT_algorithm.txt:
      1. Initialise tree T with x_start.
      2. While |T| < max_tree_size:
         3. x_samp  <- sample from X (uniform, with optional goal bias).
         4. x_nearest <- nearest node in T to x_samp (Euclidean distance).
         5. x_new   <- steer from x_nearest toward x_samp by at most step_size.
         6. If segment x_nearest -> x_new is collision-free:
            7. Add x_new to T with edge (x_nearest, x_new).
            8. If x_new in X_goal:
               9. Return SUCCESS and the path to x_new.
      13. Return FAILURE.

    Parameters
    ----------
    start_x, start_y : float
        Coordinates of the start configuration x_start.
    goal_x, goal_y : float
        Coordinates of the goal configuration.
    obstacles : list of tuple
        Circular obstacles as (cx, cy, radius) from read_obstacles().
    step_size : float, optional
        Maximum distance the local planner advances per iteration (default 0.05).
    goal_tolerance : float, optional
        Radius of the goal region X_goal; a new node is considered to have
        reached the goal when its distance to the goal is <= goal_tolerance
        (default 0.05).
    max_tree_size : int, optional
        Upper bound on the number of nodes in T before declaring failure
        (default 5 000).
    goal_bias : float, optional
        Probability of sampling the goal position directly instead of a
        uniformly random point (default 0.1 = 10 %).
    x_min, x_max : float, optional
        Horizontal bounds of the uniform sampling region (default -0.5 / 0.5).
    y_min, y_max : float, optional
        Vertical bounds of the uniform sampling region (default -0.5 / 0.5).
    seed : int or None, optional
        Random seed for reproducibility.  Pass None for a non-deterministic run.

    Returns
    -------
    tree_nodes : dict
        All nodes added to the tree: {node_id (int): {'x': float, 'y': float}}.
        Node ID 1 is always the start node.
    edges : list of tuple
        All tree edges as (parent_id, child_id, cost) where cost is the
        Euclidean distance between the two nodes.
    path : list of int or None
        Ordered node IDs from start to the goal node if a path was found,
        or None if the tree reached max_tree_size without entering X_goal.
    """
    if seed is not None:
        random.seed(seed)

    # Step 1: initialise the search tree T with x_start (node ID = 1)
    tree_nodes: dict[int, dict[str, float]] = {
        1: {'x': start_x, 'y': start_y}
    }
    parent: dict[int, Optional[int]] = {1: None}
    edges: list[tuple[int, int, float]] = []
    next_id = 2          # next available node ID
    goal_node_id: Optional[int] = None

    # Step 2: main loop - grow the tree until it reaches the goal or max size
    while len(tree_nodes) < max_tree_size:

        # Step 3: x_samp - sample a configuration from X
        x_samp, y_samp = sample_random_point(
            x_min, x_max, y_min, y_max, goal_x, goal_y, goal_bias
        )

        # Step 4: x_nearest - find the tree node closest to x_samp
        nearest_id = find_nearest_node(tree_nodes, x_samp, y_samp)
        x_near = tree_nodes[nearest_id]['x']
        y_near = tree_nodes[nearest_id]['y']

        # Step 5: local planner - advance step_size from x_nearest toward x_samp
        x_new, y_new = steer(x_near, y_near, x_samp, y_samp, step_size)

        # Step 6: check if the motion from x_nearest to x_new is collision-free
        if is_motion_collision_free(x_near, y_near, x_new, y_new, obstacles):

            # Step 7: add x_new to T with an edge from x_nearest to x_new
            node_id = next_id
            next_id += 1
            tree_nodes[node_id] = {'x': x_new, 'y': y_new}
            parent[node_id] = nearest_id
            cost = euclidean_distance(x_near, y_near, x_new, y_new)
            edges.append((nearest_id, node_id, cost))

            # Step 8: check if x_new is inside the goal region X_goal
            if is_in_goal_region(x_new, y_new, goal_x, goal_y, goal_tolerance):
                # Step 9: SUCCESS - record goal node and exit loop
                goal_node_id = node_id
                break

    # Reconstruct path from start to goal by tracing parent pointers backward
    path: Optional[list[int]] = None
    if goal_node_id is not None:
        path = []
        node: Optional[int] = goal_node_id
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

    # Step 13: return tree, edges, and path (None = FAILURE)
    return tree_nodes, edges, path


# ── output writers ────────────────────────────────────────────────────────────


def write_nodes_csv(
    filepath: str,
    tree_nodes: dict[int, dict[str, float]],
    goal_x: float,
    goal_y: float,
) -> None:
    """
    Write the RRT tree nodes to a nodes.csv file.

    Each row follows the format  ID,x,y,heuristic-cost-to-go  where the
    heuristic is the Euclidean distance from the node to the goal position,
    matching the convention used by the A* visualiser.

    Parameters
    ----------
    filepath : str
        Output path (created or overwritten).
    tree_nodes : dict
        Node dictionary as returned by rrt(): {node_id: {'x': float, 'y': float}}.
    goal_x, goal_y : float
        Goal coordinates used to compute the heuristic value for each node.

    Returns
    -------
    None
    """
    with open(filepath, 'w') as f:
        f.write('# nodes.csv - generated by chapter_10_rrt.py\n')
        f.write('# ID,x,y,heuristic-cost-to-go\n')
        for node_id in sorted(tree_nodes):
            x = tree_nodes[node_id]['x']
            y = tree_nodes[node_id]['y']
            # Heuristic = Euclidean distance from this node to the goal
            h = euclidean_distance(x, y, goal_x, goal_y)
            f.write(f'{node_id},{x:.6f},{y:.6f},{h:.6f}\n')


def write_edges_csv(
    filepath: str,
    edges: list[tuple[int, int, float]],
) -> None:
    """
    Write the RRT tree edges to an edges.csv file.

    Each row has the form  ID1,ID2,cost  where cost is the Euclidean distance
    between the two endpoints (used as the traversal cost).

    Parameters
    ----------
    filepath : str
        Output path (created or overwritten).
    edges : list of tuple
        Edge list as returned by rrt(): [(parent_id, child_id, cost), ...].

    Returns
    -------
    None
    """
    with open(filepath, 'w') as f:
        f.write('# edges.csv - generated by chapter_10_rrt.py\n')
        f.write('# ID1,ID2,cost\n')
        for id1, id2, cost in edges:
            f.write(f'{id1},{id2},{cost:.6f}\n')


def write_path_csv(
    filepath: str,
    path: Optional[list[int]],
    start_id: int,
) -> None:
    """
    Write the solution path to a path.csv file.

    If a path was found, writes a single comma-separated line of node IDs from
    start to goal.  If no path was found (path is None), writes only the start
    node ID so the visualiser can still display the starting position.

    Parameters
    ----------
    filepath : str
        Output path (created or overwritten).
    path : list of int or None
        Ordered node IDs from start to goal as returned by rrt(), or None if
        the algorithm failed to reach the goal.
    start_id : int
        ID of the start node, written as the sole entry when path is None.

    Returns
    -------
    None
    """
    with open(filepath, 'w') as f:
        f.write('# path.csv - generated by chapter_10_rrt.py\n')
        if path:
            f.write(','.join(str(n) for n in path) + '\n')
        else:
            # FAILURE: only the start node is reachable
            f.write(str(start_id) + '\n')


# ── main entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """
    Run the RRT planner on the Chapter_10_RRT scene.

    Reads obstacles.csv and the initial nodes.csv (which defines the start at
    node 1 and the goal as the node with heuristic = 0), runs RRT, then
    overwrites nodes.csv, edges.csv, and path.csv with the generated tree and
    solution path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scene_dir  = os.path.join(script_dir, 'Chapter_10_RRT')

    # Read obstacles and initial start/goal positions
    obstacles    = read_obstacles   (os.path.join(scene_dir, 'obstacles.csv'))
    initial_nodes = read_initial_nodes(os.path.join(scene_dir, 'nodes.csv'))

    # The start node is node 1; the goal is the node with heuristic = 0
    start_id = 1
    goal_id  = min(initial_nodes, key=lambda n: initial_nodes[n]['h'])

    start_x = initial_nodes[start_id]['x']
    start_y = initial_nodes[start_id]['y']
    goal_x  = initial_nodes[goal_id]['x']
    goal_y  = initial_nodes[goal_id]['y']

    # RRT parameters
    STEP_SIZE     = 0.05   # maximum distance advanced per iteration
    GOAL_TOLERANCE = 0.05  # radius of the goal region X_goal
    MAX_TREE_SIZE  = 5_000  # upper bound on tree nodes before declaring failure
    GOAL_BIAS      = 0.1   # probability of sampling the goal directly (10 %)
    X_MIN, X_MAX   = -0.5, 0.5  # sampling region horizontal bounds
    Y_MIN, Y_MAX   = -0.5, 0.5  # sampling region vertical bounds
    SEED           = None  # set to an integer for a reproducible run

    print(f"Start     : node {start_id}  ({start_x}, {start_y})")
    print(f"Goal      : node {goal_id}  ({goal_x}, {goal_y})")
    print(f"Obstacles : {len(obstacles)}")
    print(f"Step size : {STEP_SIZE}  |  Goal tolerance: {GOAL_TOLERANCE}")
    print(f"Max tree  : {MAX_TREE_SIZE}  |  Goal bias: {GOAL_BIAS * 100:.0f}%")
    print()

    # Run RRT
    tree_nodes, edges, path = rrt(
        start_x, start_y,
        goal_x, goal_y,
        obstacles,
        step_size=STEP_SIZE,
        goal_tolerance=GOAL_TOLERANCE,
        max_tree_size=MAX_TREE_SIZE,
        goal_bias=GOAL_BIAS,
        x_min=X_MIN, x_max=X_MAX,
        y_min=Y_MIN, y_max=Y_MAX,
        seed=SEED,
    )

    # Write output files (overwrite the example placeholders)
    nodes_file = os.path.join(scene_dir, 'nodes.csv')
    edges_file = os.path.join(scene_dir, 'edges.csv')
    path_file  = os.path.join(scene_dir, 'path.csv')

    write_nodes_csv(nodes_file, tree_nodes, goal_x, goal_y)
    write_edges_csv(edges_file, edges)
    write_path_csv (path_file, path, start_id)

    # Report results
    if path:
        total_cost = sum(
            euclidean_distance(
                tree_nodes[path[i]]['x'], tree_nodes[path[i]]['y'],
                tree_nodes[path[i + 1]]['x'], tree_nodes[path[i + 1]]['y'],
            )
            for i in range(len(path) - 1)
        )
        print(f"SUCCESS")
        print(f"Tree size : {len(tree_nodes)} nodes  |  {len(edges)} edges")
        print(f"Path      : {' -> '.join(str(n) for n in path)}")
        print(f"Path cost : {total_cost:.4f}  ({len(path) - 1} steps)")
    else:
        print(f"FAILURE: goal not reached within {MAX_TREE_SIZE} nodes.")
        print(f"Tree size : {len(tree_nodes)} nodes")

    print(f"\nOutput written to:")
    print(f"  {nodes_file}")
    print(f"  {edges_file}")
    print(f"  {path_file}")


if __name__ == '__main__':
    main()
