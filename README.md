# Protein Folding Simulation (2D Lattice Monte Carlo)

**Course:** Fundamentals of Biophysical Chemistry — Department of Biotechnology, IIT Madras
**Authors:** Alan Royce Gabriel (BS22B001), Guka Vignesh (BS22B016), Sairam (BS22B031)

## Overview

This project simulates the folding of a **4-bead polymer chain** on a 2D lattice using a
**Monte Carlo (Metropolis criterion) algorithm**, visualized with `pygame`. Each bead
occupies a lattice point, and the polymer evolves through **end moves** and **corner moves**,
accepted or rejected based on the change in interaction energy between beads (an HP-model-style
lattice protein folding approach).

Three variants of the interaction-energy rule were implemented and compared:

- **Q1** — Only beads of the *same type* (matching index) interact when adjacent.
- **Q2** — All non-empty adjacent beads interact, with an extra energy bonus for beads 2 and 3 (the "middle" beads).
- **Q3** — Interaction depends on adjacency to bead type 0/1 with a not-equal condition (asymmetric interaction rule).

Simulations were run at interaction energies **E = -1, -2, -5** to study how interaction
strength affects folding stability and compactness.

> **Note:** This README was reconstructed from the project writeup/report PDF (no original
> `.py` source files were available). Global constants referenced in the code
> (`WIDTH`, `HEIGHT`, `GAP_X`, `GAP_Y`, `NUM_POINTS_X`, `NUM_POINTS_Y`, `E`, `BEAD`, `BLACK`,
> `point_list`, `grid`, `polymers`) are not explicitly defined in the source report and will
> need to be reconstructed/declared before this code can run (e.g. `pygame` window size,
> lattice spacing, number of lattice points, bead colors, and the shared `point_list` /
> `grid` / `polymers` data structures used throughout).

## Requirements

```bash
pip install pygame numpy
```

## Algorithm

### 1. Creating the lattice

Builds a 2D grid of lattice points spaced `GAP_X` / `GAP_Y` apart, stored in `point_list`.

```python
def lattice():
    i = 0
    for x in range(GAP_X, WIDTH - GAP_X + 1, GAP_X):
        j = 0
        for y in range(GAP_Y, HEIGHT - GAP_Y + 1, GAP_Y):
            point_list[j][i][0] = x
            point_list[j][i][1] = y
            j += 1
        i += 1
```

### 2. Visualizing the lattice

Draws each lattice point as a small circle.

```python
def draw_lattice(surface, color, radius=3):
    for points in point_list:
        for point in points:
            # Draw each point as a circle on the surface
            pygame.draw.circle(surface, color, point, radius)
```

### 3. Creating the four-bead polymer

Randomly places a 4-bead polymer chain per row, and marks the corresponding `grid` cells
with bead indices 1–4 (0 = empty space).

```python
def polymer():
    for i in range(NUM_POINTS_Y):
        rand = random.randint(0, NUM_POINTS_X - 4)
        k = 0
        polymers[i] = [(i, rand), (i, rand + 1), (i, rand + 2), (i, rand + 3)]
        for j in range(rand, rand + 4):
            grid[j][i] = k + 1
            k += 1
```

### 4. Visualizing the polymer

Draws each bead as a colored circle (`BEAD[k]`) and connects consecutive beads with a line.

```python
def draw_polymer(polymer_lattice, surface, radius=8):
    j = 0
    for key in polymers:
        k = 0
        for i in range(len(polymers[key])):
            row = polymers[key][i][0]
            col = polymers[key][i][1]
            pygame.draw.circle(surface, BEAD[k],
                                (polymer_lattice[row][col][0],
                                 polymer_lattice[row][col][1]),
                                radius)
            if k < 3:
                row1 = polymers[key][i + 1][0]
                col1 = polymers[key][i + 1][1]
                pygame.draw.line(surface, BLACK,
                                  (polymer_lattice[row][col][0],
                                   polymer_lattice[row][col][1]),
                                  (polymer_lattice[row1][col1][0],
                                   polymer_lattice[row1][col1][1]))
            k += 1
        j += 1
```

### 5. Interaction energy calculation

Takes a `grid` array, a bead's coordinate, and its index, and sums the interaction energy `E`
contributed by its 4 lattice neighbors. The neighbor-matching condition differs per question.

**Q1 — same-type interaction only**

```python
def check_directions1(array, coord, poly, ind):
    energy1 = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        r, c = coord[0] + dr, coord[1] + dc
        if 0 <= r < len(polymers) and 0 <= c < 7:
            if array[r][c] == array[coord[0]][coord[1]]:
                energy1 += E
    return energy1
```

**Q2 — all non-empty neighbors interact, extra bonus for middle beads**

```python
def check_directions1(array, coord, poly, ind):
    energy1 = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        r, c = coord[0] + dr, coord[1] + dc
        if 0 <= r < len(polymers) and 0 <= c < 7:
            if array[r][c] != 0 and array[coord[0]][coord[1]] != 0:
                energy1 += E
                if ind[1] == 1 or ind[1] == 2:
                    energy1 -= 2 * E
                else:
                    energy1 -= E
    return energy1
```

**Q3 — asymmetric interaction rule (bead types 0/1)**

```python
def check_directions1(array, coord, poly, ind):
    energy1 = 0
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
        r, c = coord[0] + dr, coord[1] + dc
        if 0 <= r < len(polymers) and 0 <= c < 7:
            if array[r][c] != array[coord[0]][coord[1]] and (
                array[coord[0]][coord[1]] == 0 or
                array[coord[0]][coord[1]] == 1
            ):
                energy1 += E
    return energy1
```

### 6. End moves

For the two terminal beads (first and last of the chain), checks the direction vector to the
adjacent bead, then looks for empty diagonal/adjacent lattice sites the end bead could pivot into.

```python
def endmove(gridT):
    empty = {}
    for i in range(NUM_POINTS_Y):
        dir = np.array(polymers[i][1]) - np.array(polymers[i][0])

        if dir[0] == 0 and dir[1] == 1:
            # top right
            m = np.array(polymers[i][0]) + np.array((-1, 0))
            l = np.array(polymers[i][0]) + np.array((-1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))
            # bottom right
            m = np.array(polymers[i][0]) + np.array((1, 0))
            l = np.array(polymers[i][0]) + np.array((1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))

        elif dir[0] == 0 and dir[1] == -1:
            # top left
            m = np.array(polymers[i][0]) + np.array((-1, 0))
            l = np.array(polymers[i][0]) + np.array((-1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))
            # bottom left
            m = np.array(polymers[i][0]) + np.array((1, 0))
            l = np.array(polymers[i][0]) + np.array((1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))

        elif dir[0] == 1 and dir[1] == 0:
            # left bottom
            m = np.array(polymers[i][0]) + np.array((0, -1))
            l = np.array(polymers[i][0]) + np.array((1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))
            # right bottom
            m = np.array(polymers[i][0]) + np.array((0, 1))
            l = np.array(polymers[i][0]) + np.array((1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))

        elif dir[0] == -1 and dir[1] == 0:
            # left top
            m = np.array(polymers[i][0]) + np.array((0, -1))
            l = np.array(polymers[i][0]) + np.array((-1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))
            # right top
            m = np.array(polymers[i][0]) + np.array((0, 1))
            l = np.array(polymers[i][0]) + np.array((-1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, 0) not in empty:
                        empty[(i, 0)] = [(l[0], l[1])]
                    else:
                        empty[(i, 0)].append((l[0], l[1]))

        # --- same logic repeated for the OTHER end of the chain (bead index -1) ---
        dir1 = np.array(polymers[i][3]) - np.array(polymers[i][2])
        if dir1[0] == 0 and dir1[1] == 1:
            # top left
            m = np.array(polymers[i][-1]) + np.array((-1, 0))
            l = np.array(polymers[i][-1]) + np.array((-1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))
            # bottom left
            m = np.array(polymers[i][-1]) + np.array((1, 0))
            l = np.array(polymers[i][-1]) + np.array((1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))

        elif dir1[0] == 0 and dir1[1] == -1:
            # top right
            m = np.array(polymers[i][-1]) + np.array((-1, 0))
            l = np.array(polymers[i][-1]) + np.array((-1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))
            # bottom right
            m = np.array(polymers[i][-1]) + np.array((1, 0))
            l = np.array(polymers[i][-1]) + np.array((1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))

        elif dir1[0] == -1 and dir1[1] == 0:
            # left bottom
            m = np.array(polymers[i][-1]) + np.array((0, -1))
            l = np.array(polymers[i][-1]) + np.array((1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))
            # right bottom
            m = np.array(polymers[i][-1]) + np.array((0, 1))
            l = np.array(polymers[i][-1]) + np.array((1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))

        elif dir1[0] == 1 and dir1[1] == 0:
            # left top
            m = np.array(polymers[i][-1]) + np.array((0, -1))
            l = np.array(polymers[i][-1]) + np.array((-1, -1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))
            # right top
            m = np.array(polymers[i][-1]) + np.array((0, 1))
            l = np.array(polymers[i][-1]) + np.array((-1, 1))
            if 0 <= l[0] < gridT.shape[0] and 0 <= l[1] < gridT.shape[1]:
                if gridT[m[0]][m[1]] == 0 and gridT[l[0]][l[1]] == 0:
                    if (i, -1) not in empty:
                        empty[(i, -1)] = [(l[0], l[1])]
                    else:
                        empty[(i, -1)].append((l[0], l[1]))

    return empty
```

*Logic:* for bead 1 and bead 2, take the difference in their positions to get the direction
(vertical/horizontal, positive/negative), then check which diagonal/adjacent lattice cells
around bead 1 are empty; if empty, record them as candidate moves in the `empty` dict. The
same logic (with a different direction check) applies symmetrically to beads 3 and 4 at the
other end of the chain.

### 7. Corner moves

For the two "middle" beads (2 and 3), checks whether the bond vectors on either side of the
bead are perpendicular (a right-angle "kink" in the chain); if so, that bead may pivot to the
diagonally-opposite empty lattice site.

```python
def cornermove(gridT):
    empty = {}
    for i in range(NUM_POINTS_Y):
        # second bead
        dir1 = np.array(polymers[i][1]) - np.array(polymers[i][0])
        dir2 = np.array(polymers[i][2]) - np.array(polymers[i][1])
        if dir1.dot(dir2) == 0:
            if dir1[1] == 1 and dir2[0] == -1:
                if gridT[polymers[i][1][0] - 1][polymers[i][1][1] - 1] == 0:
                    if (i, 1) not in empty:
                        empty[(i, 1)] = [(polymers[i][1][0] - 1, polymers[i][1][1] - 1)]
                    else:
                        empty[(i, 1)].append((polymers[i][1][0] - 1, polymers[i][1][1] - 1))
            if dir1[0] == -1 and dir2[1] == 1:
                if gridT[polymers[i][1][0] + 1][polymers[i][1][1] + 1] == 0:
                    if (i, 1) not in empty:
                        empty[(i, 1)] = [(polymers[i][1][0] + 1, polymers[i][1][1] + 1)]
                    else:
                        empty[(i, 1)].append((polymers[i][1][0] + 1, polymers[i][1][1] + 1))
            if dir1[0] == 1 and dir2[1] == 1:
                if gridT[polymers[i][1][0] - 1][polymers[i][1][1] + 1] == 0:
                    if (i, 1) not in empty:
                        empty[(i, 1)] = [(polymers[i][1][0] - 1, polymers[i][1][1] + 1)]
                    else:
                        empty[(i, 1)].append((polymers[i][1][0] - 1, polymers[i][1][1] + 1))
            if dir1[1] == 1 and dir2[0] == 1:
                if gridT[polymers[i][1][0] + 1][polymers[i][1][1] - 1] == 0:
                    if (i, 1) not in empty:
                        empty[(i, 1)] = [(polymers[i][1][0] + 1, polymers[i][1][1] - 1)]
                    else:
                        empty[(i, 1)].append((polymers[i][1][0] + 1, polymers[i][1][1] - 1))

        # third bead
        dir1 = np.array(polymers[i][2]) - np.array(polymers[i][1])
        dir2 = np.array(polymers[i][3]) - np.array(polymers[i][2])
        if dir1.dot(dir2) == 0:
            if dir1[1] == 1 and dir2[0] == -1:
                if gridT[polymers[i][2][0] - 1][polymers[i][2][1] - 1] == 0:
                    if (i, 2) not in empty:
                        empty[(i, 2)] = [(polymers[i][2][0] - 1, polymers[i][2][1] - 1)]
                    else:
                        empty[(i, 2)].append((polymers[i][2][0] - 1, polymers[i][2][1] - 1))
            if dir1[0] == -1 and dir2[1] == 1:
                if gridT[polymers[i][2][0] + 1][polymers[i][2][1] + 1] == 0:
                    if (i, 2) not in empty:
                        empty[(i, 2)] = [(polymers[i][2][0] + 1, polymers[i][2][1] + 1)]
                    else:
                        empty[(i, 2)].append((polymers[i][2][0] + 1, polymers[i][2][1] + 1))
            if dir1[0] == 1 and dir2[1] == 1:
                if (0 < polymers[i][2][0] - 1 < gridT.shape[0]
                        and 0 < polymers[i][2][1] + 1 < gridT.shape[1]):
                    if gridT[polymers[i][2][0] - 1][polymers[i][2][1] + 1] == 0:
                        if (i, 2) not in empty:
                            empty[(i, 2)] = [(polymers[i][2][0] - 1, polymers[i][2][1] + 1)]
                        else:
                            empty[(i, 2)].append((polymers[i][2][0] - 1, polymers[i][2][1] + 1))
            if dir1[1] == 1 and dir2[0] == 1:
                if gridT[polymers[i][2][0] + 1][polymers[i][2][1] - 1] == 0:
                    if (i, 2) not in empty:
                        empty[(i, 2)] = [(polymers[i][2][0] + 1, polymers[i][2][1] - 1)]
                    else:
                        empty[(i, 2)].append((polymers[i][2][0] + 1, polymers[i][2][1] - 1))

    return empty
```

*Logic:* check whether the bond vectors before and after a bead are perpendicular
(dot product = 0). If so, there is no straight-line move available; instead check the
diagonally-opposite lattice site for a corner ("kink") move. Same logic used for bead 3,
with the roles reversed.

### 8. Metropolis Monte Carlo acceptance step

Chooses a random candidate move from all available end/corner moves, computes the energy
before/after, and applies the **Metropolis criterion**:

```python
if empty != {}:
    residue, bead = random.choice(list(empty.items()))
    bead = bead[0]
    ind = polymers[residue[0]][residue[1]]

    # move
    if residue[1] == -1:
        residue1 = (residue[0], 3)
    else:
        residue1 = residue
    residue1 = (2, 3)

    grid1 = copy.deepcopy(gridT)
    polymer1 = copy.deepcopy(polymers)
    polymer1[residue1[0]][residue1[1]] = empty[residue][0]
    grid1[bead[0]][bead[1]] = gridT[ind[0]][ind[1]]
    grid1[ind[0]][ind[1]] = 0

    energy2 = check_directions1(gridT, polymers[residue1[0]][residue1[1]], polymers, residue1)
    energy1 = check_directions1(grid1, polymer1[residue1[0]][residue1[1]], polymer1, residue1)

    w = np.exp(-(energy1 - energy2))

    if w > 1:
        polymers[residue[0]][residue[1]] = empty[residue][0]
        gridT[bead[0]][bead[1]] = gridT[ind[0]][ind[1]]
        gridT[ind[0]][ind[1]] = 0
    else:
        rand = random.random()
        if rand < w:
            polymers[residue[0]][residue[1]] = empty[residue][0]
            gridT[bead[0]][bead[1]] = gridT[ind[0]][ind[1]]
            gridT[ind[0]][ind[1]] = 0
```

*Logic:* a move is randomly selected from the available end/corner moves. The change in
energy `ΔE` is computed and the Metropolis weight

  w = exp(−ΔE / kT)   (with kT = 1 here)

is calculated. If **w > 1** (i.e. the move lowers energy or is neutral), the move is always
accepted. If **w < 1**, the move is accepted only if a uniform random number in `[0, 1)` is
less than `w`; otherwise the polymer configuration is left unchanged.

## Results Summary

Simulations were run for three interaction-energy rules (Q1, Q2, Q3), each at
E = −1, −2, and −5, tracking the polymer configuration over ~2000 iterations.

- **Q1 (same-type interaction only):** As |E| increases, folded/compact states become
  markedly more stable — once folded, the polymer resists unfolding, since only
  like-type beads contribute favorable interaction energy and act as pivots for
  end/corner moves.
- **Q2 (all-neighbor interaction, weighted for middle beads):** Produces much more
  compact/globular configurations because every bead pair (not just matching types)
  contributes favorable energy; this is most pronounced at E = −5.
- **Q3 (asymmetric 0/1-type interaction):** Beads with interaction energy (1 and 2) show
  far fewer accepted moves — they stay largely fixed as favorable contacts are preserved —
  while beads 3 and 4, which have no interaction energy, move freely on essentially every
  available step.

## Repository Structure (suggested)

```
protein-folding-simulation/
├── README.md
├── requirements.txt        # pygame, numpy
├── src/
│   ├── lattice.py           # lattice() / draw_lattice()
│   ├── polymer.py           # polymer() / draw_polymer()
│   ├── energy.py            # check_directions1() variants (Q1/Q2/Q3)
│   ├── moves.py             # endmove() / cornermove()
│   ├── metropolis.py        # Metropolis acceptance step
│   └── main.py               # pygame event loop tying it all together
└── report/
    └── Fundamentals_of_Biophysical_Chemistry.pdf
```

## Reference

Fundamentals of Biophysical Chemistry — Course Project Report, Department of Biotechnology,
IIT Madras. Alan Royce Gabriel, Guka Vignesh, Sairam.
