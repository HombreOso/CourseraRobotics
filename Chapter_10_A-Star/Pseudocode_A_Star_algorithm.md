1: OPEN ← {q_start}

2: past_cost[q_start] ← 0

3: counter ← 1

4: while OPEN is not empty and counter < MAXCOUNT do
5:     current ← first node in OPEN, remove from OPEN

6:     if current is in the goal set then
7:         return SUCCESS and the path to current
8:     end if

9:     if current is not in a previously occupied C-space grid cell then
10:        mark grid cell occupied
11:        counter ← counter + 1

12:        for each control in the discrete control set do
13:            integrate control forward a short time ∆t from current to q_new

14:            if the path to q_new is collision-free then
15:                compute cost of the path to q_new
16:                place q_new in OPEN, sorted by cost
17:                parent[q_new] ← current
18:            end if
19:        end for
20:    end if
21: end while

22: return FAILURE