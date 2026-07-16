# 영역 할당 · 경로 계획 이식 스펙 (언어 독립)

이 문서는 SCoPP_INAIR의 핵심 두 단계 — **① 영역 할당(Area Allocation)**, **② 경로 계획
(Path Planning)** — 을 다른 프로젝트/다른 언어에서 동일하게 재구현할 수 있도록
알고리즘, 수식, 유사코드, 결정성(tie-break) 규칙을 정리한다. 이 저장소에서의 실제 이식
구현은 `mission_control` 패키지의
[`grid.py`](../ros2_ws/src/mission_control/mission_control/grid.py)(격자·perimeter 샘플),
[`area_allocation.py`](../ros2_ws/src/mission_control/mission_control/area_allocation.py)(Lloyd 클러스터링 + auction),
[`path_planning.py`](../ros2_ws/src/mission_control/mission_control/path_planning.py)(NN/TSP + A* stitch)이며,
[`mission_planner.py`](../ros2_ws/src/mission_control/mission_control/mission_planner.py)의
`plan_zones()`가 `mission_map.yaml`의 `planner: scopp`일 때 이들을 오케스트레이션한다
(`planner: simple`이면 나이브 `zone_split.py`+`coverage_plan.py`를 대신 씀).
근거 논문은 Collins et al., *Scalable Coverage Path Planning of Multi-Robot Teams for
Monitoring Non-Convex Areas* (SCoPP, arXiv:2103.14709) Sec. III-C~III-E이다.

---

## 0. 사전 준비: 입력 데이터 모델

두 알고리즘은 관심영역(AOI)을 이미 **동일 폭 정사각 셀로 격자화한 결과**를 입력으로
받는다. 다른 프로젝트에서도 아래와 동일한 자료구조를 먼저 준비해야 한다.

```text
Cell:
    id: string                      # 격자 내 고유 ID, (row, col)에서 결정적으로 생성
    row: int
    col: int
    center: (x: float, y: float)    # 미터 단위 로컬 Cartesian
    perimeter_samples: [(x, y), ...]  # 셀 네 변을 균일 간격으로 표본화한 점들

Node (= robot/드론):
    id: string
    position: (x: float, y: float)  # 시작 위치, 미터 단위

Map:
    cell_width_m: float             # W = 2 * h * tan(F/2)  (h=고도, F=카메라 FoV)
    cells: [Cell, ...]              # (row, col) 오름차순 등 "결정적 순서" 고정, AOI 내부 & no-fly 밖
    nodes: [Node, ...]
```

**중요한 불변조건**

- `cells` 배열의 순서는 항상 고정(예: row-major)이어야 한다. 이후 모든 tie-break가
  이 순서(“stable index”)에 의존한다.
- `perimeter_samples`는 셀마다 개수가 같을 필요는 없지만, 각 점이 어느 셀에서 나왔는지
  (`cell_id`)를 유지해야 한다. 원본은 각 변을 `ceil(W / spacing)`개로 나누며 기본
  `spacing = W / 8`을 쓴다.
- 격자화(no-fly zone 제외, AOI 밖 제외) 자체는 이 문서의 범위가 아니다. 이미 "유효
  셀 목록"이 있다고 가정한다.

---

## 1부. 영역 할당 (Area Allocation)

세 단계로 구성된다: **(1) Lloyd형 클러스터링 → (2) 충돌 셀 판정 → (3) 그리디 경매**.

### 1.1 1단계 — perimeter 표본점에 대한 Lloyd(K-means) 클러스터링

**입력**: 모든 유효 셀의 `perimeter_samples`를 한 곳에 모은 점 집합, 노드 시작 위치
(`node.position`)를 초기 중심(centroid)으로 사용.

**목적**: 각 표본점에 가장 가까운 노드(클러스터) 라벨을 부여하여, 이후 셀 단위 소유권
판정의 기초 자료로 쓴다. **셀 중심이 아니라 셀 둘레 표본점**에 대해 클러스터링한다는
점이 핵심이다 — 이래야 하나의 셀이 두 클러스터 경계에 걸치는지(충돌 셀 여부)를 알 수
있다.

**알고리즘 (표준 Lloyd 반복)**

```text
function lloyd_cluster(points, initial_centroids, tolerance_m, max_iterations):
    centroids = initial_centroids
    for iteration in 1..max_iterations:
        # 할당 단계
        for each point p in points:
            label[p] = argmin_i squared_distance(p, centroids[i])
            # 동점 시 낮은 클러스터 인덱스가 이긴다 (결정적)

        # 갱신 단계
        new_centroids = []
        for i in 0..n_clusters-1:
            members = points where label == i
            if members is empty:
                new_centroids[i] = centroids[i]      # 빈 클러스터는 이전 중심 유지
            else:
                new_centroids[i] = mean(members)      # 산술 평균 (면적 가중 아님)

        movement = max_i euclidean_distance(centroids[i], new_centroids[i])
        centroids = new_centroids
        if movement <= tolerance_m:
            return centroids, relabel(points, centroids), iteration, converged=True

    return centroids, relabel(points, centroids), max_iterations, converged=False
```

**기본 파라미터 (논문/원본 구현 기준값)**

| 파라미터 | 기본값 | 비고 |
|---|---|---|
| `tolerance_m` | `cell_width_m / 8` | 중심 이동량이 이 값 이하이면 수렴 |
| `max_iterations` | `10` | 논문 고정값 |
| 초기 중심 | 노드 시작 위치 그대로 | 노드 수 = 클러스터 수 |
| 거리 함수 | 유클리드 제곱거리 | `argmin`에는 제곱을 써도 결과 동일 |

**결정성 규칙**: 두 클러스터 중심까지의 거리가 정확히 같으면 **클러스터 인덱스가
작은 쪽**이 이긴다. (원본: `min(range(n), key=lambda i: (dist_i, i))`)

**대안 프로파일**: 실무에서 셀 수가 매우 많을 때는 정확한 Lloyd 대신
`MiniBatchKMeans`(scikit-learn 계열) 같은 근사 K-means를 써도 된다. 다만 이 경우
① 클러스터 중심에는 논문상 의미(“어느 노드가 어느 중심을 갖는가”)가 없으므로,
각 클러스터 중심을 **아직 배정되지 않은 노드 중 가장 가까운 노드**에 그리디로 매칭하는
후처리가 필요하고, ② 결과가 시드(seed)에 의존하므로 시드를 반드시 실험 메타데이터로
기록해야 한다.

**셀 단위 라벨 집계**: 클러스터링은 점 단위 결과이므로, 셀의 클러스터 후보 집합은
다음과 같이 만든다.

```text
for each cell c:
    cell.cluster_candidates = sorted(set(label[p] for p in c.perimeter_samples))
    cell.is_conflict = len(cell.cluster_candidates) > 1
```

- `cluster_candidates`가 1개면 그 셀은 해당 클러스터(노드)에 **바로** 배정된다
  (경매 대상 아님).
- 2개 이상이면 **충돌 셀(conflict cell)** 이며 2단계 경매로 넘어간다.

### 1.2 2단계 — 충돌 셀 처리 순서 고정

경매는 셀을 하나씩 순차 처리하며 그 처리 순서가 결과에 영향을 준다(그리디이므로).
**`cells` 배열의 원래 결정적 순서(row-major 등)** 를 그대로 유지한 채, 그 중
`is_conflict == true`인 셀만 걸러서 처리 큐를 만든다. 별도의 정렬 기준(예: 좌표순
재정렬)을 추가하면 원본과 다른 결과가 나오므로 주의한다.

### 1.3 3단계 — 그리디 충돌 셀 경매 (Greedy Conflict Auction)

**사전 계산 (경매 시작 전에 고정, 논문 Sec. III-D)**

각 노드(클러스터) `r`에 대해 "초기 접근 거리 편향(distance bias)" `d_B(r)`을 미리
한 번 계산해서 고정한다.

```text
for each node r:
    initial_cells(r) = 셀 중 cluster_candidates == [r] 인 것들 (비충돌·단독 배정 셀)
    if initial_cells(r) is empty:
        initial_cells(r) = 셀 중 r ∈ cluster_candidates 인 모든 것 (충돌 후보 포함, 폴백)
    if initial_cells(r) is still empty:
        d0(r) = 0
        continue

    d0(r) = round( min_{c ∈ initial_cells(r)} euclidean(node[r].position, c.center) / cell_width_m )
    d_B(r) = d0(r) * B          # B = auction bias, 기본값 0.5
```

- `d0(r)`은 **셀 폭 단위로 반올림한 정수 거리**다(연속 거리가 아님). 즉 "노드
  시작점에서 자신이 초기에 단독 지배하는 영역까지, 셀 몇 개 만큼 떨어져 있는가".
- 이 값은 경매 도중에는 다시 계산하지 않는다(한 번 고정).

**경매 루프 (충돌 셀을 고정 순서로 하나씩 처리)**

```text
assigned_count[r] = |initial_cells(r)|   # 각 노드가 이미 확보한 셀 수 (실시간 갱신)

for each conflict cell c in fixed_order:
    candidates = c.cluster_candidates
    bids = []
    for r in candidates:
        bid = assigned_count[r] + d_B(r)
        bids.append(bid)

    winner = candidates[ argmin over bids, tie-break by smaller cluster index r ]

    owner[c] = winner
    assigned_count[winner] += 1     # 다음 충돌 셀 경매에 즉시 반영 (그리디 online)
```

- **입찰가(bid) 공식**: `bid_r = N_r + B * d0(r)`
  - `N_r` = 노드 `r`이 *지금까지* 확보한 셀 개수(매 경매마다 증가하는 실시간 카운트).
  - `d0(r)`은 고정값(위에서 미리 계산), `B`는 편향 계수(기본 `0.5`).
  - **낮은 입찰가가 승리** — 이미 셀을 많이 가진 노드(N_r 큼)나, 초기 자기 영역에서
    멀리 시작한 노드(d0 큼)일수록 이 셀을 덜 원하게 되어, 결과적으로 부하가
    균등해지고 노드가 자기 시작점 근처 영역을 우선 갖도록 유도한다.
- **동점 처리**: 입찰가가 완전히 같으면 **클러스터 인덱스가 작은 노드**가 승리한다.
- 경매는 **온라인·그리디**다. 한 셀의 승자가 결정되면 그 노드의 `N_r`이 즉시 1
  증가하고, 다음 충돌 셀의 입찰가 계산에 반영된다. 전역 최적화(예: Hungarian
  algorithm)가 아니라 순차 그리디임에 유의한다.

**출력 계약**

```text
AllocationResult:
    owner_by_cell: {cell_id -> node_index}      # 모든 유효 셀이 정확히 하나의 소유자를 가짐
    cells_by_node: {node_index -> [cell_id, ...]}
    bias: float
    d0_by_node: [float, ...]                     # 진단/재현용으로 보존 권장
```

**검증 불변조건**

- 모든 유효 셀은 정확히 하나의 소유자를 갖는다.
- 비충돌 셀의 소유자는 클러스터링 결과와 100% 일치한다.
- 동일 입력 + 동일 시드로 항상 동일한 `owner_by_cell`을 재생산한다(완전 결정적).

---

## 2부. 경로 계획 (Path Planning)

영역 할당이 끝나면 각 노드는 자신이 소유한 셀 목록을 갖는다. 경로 계획은 **(a) 방문
순서 결정 → (b) 격자 인접 경로로 스티칭(연결) → (c) 왕복 궤적/거리 계산** 3단계로
이뤄진다. 두 가지 프로파일을 지원한다: **`paper_nn`**(논문 원안, KD-tree 그리디
최근접 이웃)과 **`metric_tsp`**(확장 옵션, 그래프 최단거리 기반 TSP).

### 2.0 공통 준비물

- `stable_index[cell_id]`: 전체 유효 셀 배열에서의 인덱스. 모든 동점 처리에 사용.
- `id_by_key[(row, col)] = cell_id`: 4-이웃 탐색용 격자 인접 조회 테이블.
- **주의**: no-fly/AOI 밖 셀은 애초에 `cells` 목록에 없으므로, 이 인접 경로 탐색은
  자연스럽게 장애물을 우회한다(단, 우회 경로가 아예 없으면 실패 처리, 2.3절 참고).

### 2.1 4-이웃 그리드 경로 스티칭 (두 프로파일 공통 서브루틴)

두 목표 셀 사이를 실제로 "이동 가능한" 경로로 잇는 서브루틴. **A\* (4-방향 이동,
균일 비용 1, 맨해튼 휴리스틱)** 를 사용한다.

```text
function adjacent_path(start_id, goal_id):
    if start_id == goal_id: return [start_id]

    start_key = (start.row, start.col)
    goal_key  = (goal.row, goal.col)
    frontier = priority_queue()  # (f = g + h, g, insertion_seq, node_key)
    push(frontier, (0, 0, 0, start_key))
    came_from = {start_key: null}
    cost = {start_key: 0}
    seq = 0

    while frontier not empty:
        (_, g, _, current) = pop_min(frontier)
        if current == goal_key:
            return reconstruct_path(came_from, current)   # id 목록, 시작~끝 포함
        if g != cost[current]: continue   # 오래된 큐 항목 skip

        for (dr, dc) in [(-1,0), (0,-1), (0,1), (1,0)]:   # 상, 좌, 우, 하 순서 고정
            neighbor = (current.row+dr, current.col+dc)
            if neighbor not in valid_cells: continue
            new_g = g + 1
            if new_g < cost.get(neighbor, INF):
                cost[neighbor] = new_g
                came_from[neighbor] = current
                seq += 1
                h = |neighbor.row - goal.row| + |neighbor.col - goal.col|
                push(frontier, (new_g + h, new_g, seq, neighbor))

    raise NoAdjacentPathError   # 두 셀 사이에 유효 경로가 없음
```

- 이웃 탐색 순서(`(-1,0),(0,-1),(0,1),(1,0)`)와 `insertion_seq` 를 우선순위 큐 tie-break
  에 넣는 이유는 **f값이 같은 여러 경로 중 항상 동일한 하나**를 고르기 위함이다
  (재현성). 다른 언어로 옮길 때도 동일한 이웃 순회 순서 + FIFO 안정 정렬을 유지해야
  결과가 일치한다.
- 시작 셀 결정: 노드의 실제 시작 위치(임의의 연속 좌표)를 격자에 스냅할 때는
  **가장 가까운 셀 중심**을 고른다. 동점이면 `stable_index`가 작은 셀.

```text
function start_cell_id(node_position):
    return argmin over all valid cells c of (squared_distance(c.center, node_position), stable_index[c])
```

### 2.2 프로파일 A — `paper_nn` (논문 원안: KD-tree 그리디 최근접 이웃)

**목표 순서 결정** — 소유한 셀 중심들을 노드 시작 위치에서부터 그리디하게 가장 가까운
순서로 방문한다.

```text
function ordered_nearest_neighbor(start_position, owned_cells):
    remaining = owned_cells
    current = start_position
    route = []
    while remaining not empty:
        tree = build_kdtree(remaining)     # 매 반복마다 재구축 (lazy deletion 아님)
        chosen = nearest(tree, current)
        route.append(chosen)
        current = chosen.center
        remaining = remaining - {chosen}
    return route
```

**KD-tree 세부사항 (2차원, 결정적)**

```text
function build_kdtree(items, depth=0):
    if items empty: return null
    axis = depth % 2                      # 0: x, 1: y 번갈아 분할
    sorted_items = sort(items, key=(item.point[axis], item.point[1-axis], item.stable_index))
    mid = len(sorted_items) // 2
    return Node(
        item = sorted_items[mid],
        axis = axis,
        left  = build_kdtree(sorted_items[:mid], depth+1),
        right = build_kdtree(sorted_items[mid+1:], depth+1),
    )

function nearest(root, target):
    best = root.item
    best_key = (squared_distance(root.item.point, target), root.item.stable_index)

    function visit(node):
        if node is null: return
        key = (squared_distance(node.item.point, target), node.item.stable_index)
        if key < best_key: best, best_key = node.item, key
        delta = target[node.axis] - node.item.point[node.axis]
        near, far = (node.left, node.right) if delta <= 0 else (node.right, node.left)
        visit(near)
        if delta^2 <= best_key[0]:        # 반대편 서브트리 가지치기 조건
            visit(far)

    visit(root)
    return best
```

- 동점(거리 완전히 동일) 시 `stable_index`(전역 셀 배열 순서)가 작은 쪽이 이긴다.

**실제 이동 경로(스티칭) 생성**

`ordered_nearest_neighbor`가 만든 것은 "방문 순서"일 뿐, 셀 중심 사이를 직선으로
잇는 것이 아니라 **격자 인접 경로**로 실제 이동 궤적을 만든다.

```text
function plan_paper_nn_path(node):
    ordered = ordered_nearest_neighbor(node.position, node.owned_cells)
    if ordered is empty: return empty_path(node)

    start_cell = start_cell_id(node.position)          # 2.1절
    motion_ids = [start_cell]
    current = start_cell
    for target in ordered:
        segment = adjacent_path(current, target.cell_id)   # 2.1절, 시작 셀 포함
        motion_ids += segment[1:]                            # 중복 시작점 제거하고 이어붙임
        current = target.cell_id

    return_motion_index = len(motion_ids)               # 여기부터 "귀환" 구간
    motion_ids += adjacent_path(current, start_cell)[1:] # 시작 셀로 복귀

    motion_waypoints = [cell_center(id) for id in motion_ids]
    distance = euclidean(node.position, motion_waypoints[0])
             + sum(euclidean(a, b) for a, b in consecutive_pairs(motion_waypoints))
             + euclidean(motion_waypoints[-1], node.position)

    trajectory = [node.position] + motion_waypoints + ([node.position] if motion_waypoints else [])
    return NodePath(ordered_cell_ids, motion_ids, motion_waypoints, return_motion_index, distance, trajectory)
```

### 2.3 프로파일 B — `metric_tsp` (확장 옵션: 그래프 최단거리 기반 TSP)

`paper_nn`은 그리디이므로 최적이 아니다. 더 짧은 총 이동거리가 필요하면 "격자 그래프
최단거리"를 메트릭으로 쓰는 TSP로 방문 순서를 최적화할 수 있다.

**1) 거리 행렬(metric closure) 구성**

```text
depot = start_cell_id(node.position)
targets = sort(node.owned_cells, key=stable_index)     # 결정적 순서로 정렬
metric_nodes = [depot] + targets

for i, u in metric_nodes:
    for j, v in metric_nodes:
        if i == j: D[i][j] = 0
        else: D[i][j] = (len(adjacent_path(u, v)) - 1) * cell_width_m
```

`adjacent_path`(2.1절 A\*)의 홉(hop) 수 × 셀 폭이 그래프상의 실제 이동 거리다
(장애물/no-fly 우회를 반영한 진짜 최단 경로 길이).

**2) 방문 순서 최적화 — 목표 수에 따라 정확해/휴리스틱 분기**

```text
function metric_tsp_order(D):
    n = len(D) - 1   # depot 제외 목표 수
    if n <= 20:
        return held_karp_cycle(D)          # 정확해, O(n^2 * 2^n)
    else:
        return insertion_two_opt_cycle(D)  # 휴리스틱, 대규모 대응
```

**2-a) 정확해: Held-Karp DP (목표 ≤ 20)**

depot(인덱스 0)에서 출발해 모든 목표를 정확히 한 번 방문하고 depot으로 돌아오는
최소 비용 해밀턴 사이클을 비트마스크 DP로 구한다.

```text
function held_karp_cycle(D):
    n = len(D) - 1
    # states[(mask, last)] = (최소 누적 비용, 그 경로에서 last 직전 노드)
    states = {}
    for t in 1..n:
        states[({t}, t)] = (D[0][t], 0)

    for mask in all subsets of {1..n} in increasing size order:
        for last in mask:
            if (mask, last) not in states: continue
            cost, _ = states[(mask, last)]
            for nxt in {1..n} - mask:
                new_mask = mask + {nxt}
                candidate = cost + D[last][nxt]
                if better than states.get((new_mask, nxt)):
                    states[(new_mask, nxt)] = (candidate, last)

    full = {1..n}
    best_last = argmin over last in 1..n of states[(full, last)].cost + D[last][0]
    order = backtrack states from (full, best_last) to get visiting sequence
    return order   # 목표 인덱스들의 방문 순서 (1..n 값들)
```

- 시간복잡도가 지수적이므로 **목표 20개**를 실용적 상한으로 둔다(원본 구현 기준).
  더 큰 상한이 필요하면 프로젝트 성능 예산에 맞춰 조정하되, 지수 폭증을 인지해야 한다.

**2-b) 휴리스틱: 최저-증가 삽입(cheapest insertion) + 2-opt (목표 > 20)**

```text
function insertion_two_opt_cycle(D):
    remaining = {1..n}
    first = argmin_{t in remaining} D[0][t]      # depot에서 가장 가까운 목표로 시작
    order = [first]; remaining -= {first}

    while remaining not empty:
        best = null   # (증가비용, 항목, 삽입위치)
        for item in remaining:
            for position in 0..len(order):
                prev = depot if position == 0 else order[position-1]
                next = depot if position == len(order) else order[position]
                increase = D[prev][item] + D[item][next] - D[prev][next]
                candidate = (increase, item, position)
                if best is null or candidate < best: best = candidate
        (_, item, position) = best
        insert item at position in order
        remaining -= {item}

    # 2-opt 로컬 서치 (개선이 없을 때까지, 개선 발견 즉시 재시작)
    improved = true
    while improved:
        improved = false
        for i in 0..len(order)-2:
            for j in i+1..len(order)-1:
                before = depot if i == 0 else order[i-1]
                after  = depot if j == len(order)-1 else order[j+1]
                delta = D[before][order[j]] + D[order[i]][after]
                      - D[before][order[i]] - D[order[j]][after]
                if delta < -1e-9:
                    reverse order[i..j]
                    improved = true; break
            if improved: break

    return order
```

**3) 실제 이동 경로 스티칭 & 거리 계산** — 2.2절의 스티칭 로직과 완전히 동일하다.
`ordered_ids`(TSP로 얻은 방문 순서)를 목표 시퀀스로 사용해 `adjacent_path`로
연결하고, `motion_waypoints` 사이 유클리드 거리를 합산해 `distance_m`을 얻는다.

### 2.4 오류/경계 조건

- `NoAdjacentPathError`: 두 셀 사이에 유효한 4-이웃 경로가 전혀 없을 때(예: 격자가
  분리된 두 덩어리로 나뉘어 서로 연결되지 않음). 호출자는 이를 실패로 처리하거나,
  분리된 영역 간 별도 연결 로직(이 문서 범위 밖)을 추가해야 한다.
- `MetricTspTooLargeError`(내부적으로만 사용): 정확해 상한(20)을 넘으면 자동으로
  휴리스틱으로 폴백한다. 사용자에게 노출할 필요는 없다.
- 소유 셀이 0개인 노드: 빈 경로(정지 상태)를 정상 결과로 반환한다. 예외를 던지지
  않는다.

---

## 3부. 이식 체크리스트

다른 프로젝트에 이식할 때 다음을 반드시 지켜야 원본과 동일한 결과가 나온다.

1. **셀 순회 순서를 고정**한다 (row-major 등). 모든 tie-break가 이 순서에 의존한다.
2. 클러스터링은 **셀 중심이 아니라 셀 둘레 표본점**에 대해 수행한다.
3. Lloyd 수렴 조건은 `tolerance = cell_width / 8`, `max_iterations = 10`이 기본값이다.
4. 경매 입찰가 `bid_r = N_r + B * d0(r)`에서 `d0(r)`은 **경매 시작 전 한 번만** 계산해
   고정하고, `N_r`만 경매 진행에 따라 실시간 갱신한다. 기본 `B = 0.5`.
5. 모든 동점(거리, 입찰가)은 "더 작은 인덱스가 이긴다"로 통일한다 — 클러스터 인덱스,
   셀의 stable index 등.
6. 두 셀을 잇는 실제 이동 경로는 직선이 아니라 **4-이웃 A\* 격자 경로**로 만든다.
   이웃 순회 순서(`상,좌,우,하` 등 고정 순서)와 우선순위 큐의 삽입 순서 tie-break까지
   맞춰야 완전히 동일한 경로가 나온다.
7. 경로 계획 결과의 `distance_m`은 (실제 위치 → 첫 격자 셀) + (격자 경로 셀-센터 간
   유클리드 합) + (마지막 셀 → 실제 위치 복귀)로 계산한다.
8. TSP 정확해는 목표 20개 이하에서만 쓰고, 그 이상은 삽입+2-opt 휴리스틱으로
   자동 폴백한다.
9. 랜덤 요소(예: MiniBatchKMeans 근사 클러스터링을 쓸 경우)는 항상 시드를 외부
   주입하고 결과 메타데이터에 기록한다.

## 4. 기본 파라미터 요약표

| 파라미터 | 기본값 | 위치 |
|---|---|---|
| 클러스터 수렴 허용오차 | `cell_width_m / 8` | 1.1절 |
| 클러스터 최대 반복 | `10` | 1.1절 |
| 경매 편향 계수 `B` | `0.5` | 1.3절 |
| perimeter 표본 간격 | `cell_width_m / 8` | 0절 |
| TSP 정확해 상한 (목표 수) | `20` | 2.3절 |
| 4-이웃 이동 비용 | 셀당 `1` (× `cell_width_m`) | 2.1절 |
