
  Chapter 10 - Rapidly-exploring Random Tree (RRT) Motion Planner


OVERVIEW
--------
This code implements the RRT path planning algorithm for a 2-D configuration
space.  The planner grows a tree from a start position by repeatedly sampling
random configurations and extending the tree toward them, until a node lands
inside the goal region or the tree reaches its maximum size.

The implementation lives in:
  chapter_10_rrt.py          (project root)

It reads from and writes to the scene folder:
  Chapter_10_RRT/
    obstacles.csv            (input  - do not overwrite manually)
    nodes.csv                (output - rewritten on every run)
    edges.csv                (output - rewritten on every run)
    path.csv                 (output - rewritten on every run)


HOW TO RUN
----------
From the project root directory:

  python chapter_10_rrt.py

The script prints a run summary to the console and overwrites the three output
CSV files.  Example output:

  Start     : node 1  (-0.5, -0.5)
  Goal      : node 2  (0.5, 0.5)
  Obstacles : 8
  Step size : 0.05  |  Goal tolerance: 0.05
  Max tree  : 5000  |  Goal bias: 10%

  SUCCESS
  Tree size : 84 nodes  |  83 edges
  Path      : 1 -> 2 -> 3 -> ... -> 84
  Path cost : 1.7869  (36 steps)


ALGORITHM DESCRIPTION
---------------------
The algorithm follows the pseudocode in RRT_algorithm.txt step by step:

  Step 1   Initialise tree T with x_start (node 1 at the start position).

  Step 2   Enter the main loop.  Continue while |T| < max_tree_size.

  Step 3   x_samp - sample a configuration uniformly at random from the
           square region [-0.5, 0.5] x [-0.5, 0.5].  With probability
           goal_bias (default 10%) the goal position is sampled directly
           instead, which pulls the tree toward the goal and speeds up
           convergence.

  Step 4   x_nearest - scan every node currently in T and pick the one
           with the smallest Euclidean distance to x_samp.

  Step 5   Local planner - compute x_new by advancing at most step_size
           (default 0.05) from x_nearest along the straight line toward
           x_samp.  If x_samp is already within step_size, x_new = x_samp.

  Step 6   Collision check - test whether the straight-line segment from
           x_nearest to x_new intersects any circular obstacle.  The check
           uses a parametric quadratic formula (see segment_intersects_circle).

  Step 7   If the motion is collision-free, add x_new to T as a new node and
           record the edge (x_nearest -> x_new) with cost = Euclidean distance.

  Step 8-9 If x_new lies within goal_tolerance (default 0.05) of the goal
           position, declare SUCCESS and trace parent pointers back to the
           start to reconstruct the solution path.

  Step 13  If the loop ends without reaching the goal, return FAILURE and
           write only the start node ID to path.csv.

Distance metric and edge cost: Euclidean distance is used both to identify the
nearest tree node (Step 4) and as the cost of each new edge (Step 7).


PARAMETERS (set in main() inside chapter_10_rrt.py)
----------------------------------------------------
  STEP_SIZE       0.05   Maximum distance the tree extends per iteration.
                         Smaller values navigate tighter spaces but need more
                         iterations; larger values converge faster in open space.

  GOAL_TOLERANCE  0.05   Radius of the goal region X_goal.  A new node is
                         accepted as "at the goal" when its distance to the
                         goal position is <= GOAL_TOLERANCE.

  MAX_TREE_SIZE   5000   Hard upper bound on the number of tree nodes.  The
                         algorithm declares FAILURE if this is reached before
                         finding the goal.

  GOAL_BIAS       0.1    Probability (10%) of sampling the goal directly in
                         Step 3.  Set to 0 for a purely uniform RRT.

  X_MIN / X_MAX   -0.5 / 0.5   Horizontal bounds of the sampling region.
  Y_MIN / Y_MAX   -0.5 / 0.5   Vertical bounds of the sampling region.

  SEED            None   Integer seed for the random number generator.
                         Set to any integer (e.g. 42) for a reproducible run.


FILE FORMAT (all CSV files)
---------------------------
Lines beginning with # are comments and are ignored by all readers.

  nodes.csv
    One row per tree node:  ID, x, y, heuristic-cost-to-go
      ID                    Integer node ID, 1 through N.
      x, y                  2-D coordinates of the node.
      heuristic-cost-to-go  Euclidean distance from this node to the goal
                            (admissible heuristic, compatible with A* viewer).

  edges.csv
    One row per tree edge:  ID1, ID2, cost
      ID1, ID2              Node IDs of the parent and child nodes.
      cost                  Euclidean distance between the two nodes.

  path.csv
    A single comma-separated line of node IDs from start to goal.
    If the planner failed, contains only the start node ID.

  obstacles.csv  (input, not modified by the planner)
    One row per circular obstacle:  x, y, diameter
      x, y                  Centre of the obstacle circle.
      diameter              Full width; the planner internally uses radius = diameter/2.


CODE STRUCTURE
--------------
  Shared helpers (reused from chapter_10_a_star.py)
    read_obstacles             Parse obstacles.csv into (cx, cy, radius) tuples.
    segment_intersects_circle  Parametric segment-circle intersection test.

  Input / output
    read_initial_nodes         Read the two-node start/goal seed file.
    write_nodes_csv            Write the full tree to nodes.csv.
    write_edges_csv            Write all tree edges to edges.csv.
    write_path_csv             Write the solution path (or start-only) to path.csv.

  RRT helpers
    euclidean_distance         Distance metric and edge cost.
    sample_random_point        Uniform sampling with optional goal bias.
    find_nearest_node          Linear scan for the nearest tree node.
    steer                      Local planner - advance step_size toward sample.
    is_motion_collision_free   Segment collision check against all obstacles.
    is_in_goal_region          Test whether a point is inside X_goal.

  Core algorithm
    rrt                        Full RRT implementation; returns tree, edges, path.

  Entry point
    main                       Wires everything together and writes output files.