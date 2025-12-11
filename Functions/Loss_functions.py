import numpy as np


def path_full_from_inner(inner, start, goal):
    return np.vstack([start, inner.astype(np.float32), goal])


# ==============================
# Loss components and analytic gradients
# ==============================
def path_length_Loss(start, goal, inner):
    P = path_full_from_inner(inner, start, goal)
    diffs = P[1:] - P[:-1]
    loss = np.sum(np.sum(diffs**2, axis=1))
    grad_full = np.zeros_like(P)
    seg_contrib = 2 * diffs
    grad_full[:-1] -= seg_contrib
    grad_full[1:] += seg_contrib
    return loss, grad_full[1:-1]

def path_smoothness_Loss(start, goal, inner):
    P = path_full_from_inner(inner, start, goal)
    q = P[2:] - 2 * P[1:-1] + P[:-2]
    loss = np.sum(np.sum(q**2, axis=1))
    grad_full = np.zeros_like(P)
    grad_full[:-2] += q
    grad_full[1:-1] += -2*q
    grad_full[2:] += q
    grad_full *= 2
    return loss, grad_full[1:-1]

def path_obstacle_Loss(start, goal, inner, obstacles, sample_pts_base=4, margin=2.0, threshold_dist=5.0):
    P = path_full_from_inner(inner, start, goal)
    num_segments = P.shape[0] - 1
    loss = 0.0
    grad_inner = np.zeros_like(inner, dtype=np.float32)

    Pi = P[:-1]
    Pi1 = P[1:]
    segment_centers = (Pi + Pi1) / 2

    # 장애물 정보 언패킹
    x_min = obstacles[:,0] - margin
    x_max = obstacles[:,1] + margin
    y_min = obstacles[:,2] - margin
    y_max = obstacles[:,3] + margin
    cx_list = obstacles[:,4]
    cy_list = obstacles[:,5]

    # 거리 기반 필터링
    dx_min = np.maximum(x_min.reshape(-1,1) - segment_centers[:,0], 0)
    dx_max = np.maximum(segment_centers[:,0] - x_max.reshape(-1,1), 0)
    dy_min = np.maximum(y_min.reshape(-1,1) - segment_centers[:,1], 0)
    dy_max = np.maximum(segment_centers[:,1] - y_max.reshape(-1,1), 0)
    dist2 = dx_min**2 + dx_max**2 + dy_min**2 + dy_max**2

    candidate_mask = dist2 < threshold_dist**2

    ts_base = np.linspace(0.0, 1.0, sample_pts_base).reshape(-1,1)
    ts_dense = np.linspace(0.0, 1.0, sample_pts_base*2).reshape(-1,1)

    for i in range(num_segments):
        obs_idx = np.where(candidate_mask[:, i])[0]
        if obs_idx.size == 0:
            continue

        dmin = np.min(dist2[obs_idx, i])
        ts = ts_base if dmin > margin else ts_dense

        Pi_i = Pi[i]
        Pi1_i = Pi1[i]
        pts = (1 - ts) * Pi_i + ts * Pi1_i

        px = pts[:,0]
        py = pts[:,1]

        for j in obs_idx:
            xm, xx, ym, yx = x_min[j], x_max[j], y_min[j], y_max[j]
            cx, cy = cx_list[j], cy_list[j]

            # 1. 사각형 내부인지 확인
            mask = (px >= xm) & (px <= xx) & (py >= ym) & (py <= yx)
            if not np.any(mask):
                continue

            # 유효한 포인트들만 추출
            px_m = px[mask]
            py_m = py[mask]

            # 비대칭 정규화
            w_left = cx - xm
            w_right = xx - cx
            h_down = cy - ym
            h_up = yx - cy

            dx_div = np.where(px_m < cx, w_left, w_right)
            dy_div = np.where(py_m < cy, h_down, h_up)

            nx = (px_m - cx) / dx_div
            ny = (py_m - cy) / dy_div

            # 안전장치 (클리핑)
            nx = np.clip(nx, -1.0, 1.0)
            ny = np.clip(ny, -1.0, 1.0)

            # X축 성분
            exp_x = np.exp(-nx**2)
            term_x_base = (1 - nx**2)
            term_x = term_x_base**2
            val_x = exp_x * term_x
            
            # Y축 성분
            exp_y = np.exp(-ny**2)
            term_y_base = (1 - ny**2)
            term_y = term_y_base**2
            val_y = exp_y * term_y

            # 최종 포텐셜
            A = 8.0
            V = A * val_x * val_y
            
            loss += np.sum(V)

            # Gradient 계산
            dF_dnx = -2 * nx * exp_x * term_x_base * (3 - nx**2)
            dF_dny = -2 * ny * exp_y * term_y_base * (3 - ny**2)

            # Chain Rule
            dV_dnx = A * dF_dnx * val_y
            dV_dny = A * val_x * dF_dny

            grad_px_local = dV_dnx / dx_div
            grad_py_local = dV_dny / dy_div

            # ==========================================================
            # [수정] 장애물 중심(Local Maximum) 탈출을 위한 랜덤 노이즈 추가
            # ==========================================================
            # nx, ny가 0에 가까우면(중심) gradient가 0이 되어 움직이지 않음.
            # 정규화 좌표 기준 거리 제곱
            r_sq = nx**2 + ny**2
            
            # 중심 반경 임계값 (예: 0.1 이내면 중심이라고 판단)
            center_epsilon = 0.1 
            stuck_mask = r_sq < (center_epsilon**2)
            
            if np.any(stuck_mask):
                n_stuck = np.sum(stuck_mask)
                noise_scale = 50.0 
                
                rand_kick_x = np.random.uniform(-1, 1, n_stuck) * noise_scale
                rand_kick_y = np.random.uniform(-1, 1, n_stuck) * noise_scale

                grad_px_local[stuck_mask] += rand_kick_x
                grad_py_local[stuck_mask] += rand_kick_y
            # ==========================================================

            grad_pts = np.stack([grad_px_local, grad_py_local], axis=1)

            mask_indices = np.where(mask)[0]
            ws_i = (1 - ts[mask_indices]).reshape(-1,1)
            ws_ip1 = ts[mask_indices].reshape(-1,1)

            if i > 0:
                grad_inner[i-1] += np.sum(grad_pts * ws_i, axis=0)
            if i < num_segments - 1:
                grad_inner[i] += np.sum(grad_pts * ws_ip1, axis=0)

    return loss, grad_inner




def calculate_potential_grid(x_grid, y_grid, obstacles, margin, lambda_obs):
    Z = np.zeros_like(x_grid)
    
    x_min = obstacles[:,0] - margin
    x_max = obstacles[:,1] + margin
    y_min = obstacles[:,2] - margin
    y_max = obstacles[:,3] + margin
    cx_list = obstacles[:,4]
    cy_list = obstacles[:,5]
    
    for k in range(len(obstacles)):
        xm, xx, ym, yx = x_min[k], x_max[k], y_min[k], y_max[k]
        cx, cy = cx_list[k], cy_list[k]
        
        mask = (x_grid >= xm) & (x_grid <= xx) & (y_grid >= ym) & (y_grid <= yx)
        if not np.any(mask): continue

        X_m = x_grid[mask]
        Y_m = y_grid[mask]
        
        w_left = cx - xm
        w_right = xx - cx
        h_down = cy - ym
        h_up = yx - cy
        
        dx_div = np.where(X_m < cx, w_left, w_right)
        dy_div = np.where(Y_m < cy, h_down, h_up)
        dx_div = np.maximum(dx_div, 1e-6)
        dy_div = np.maximum(dy_div, 1e-6)

        nx = (X_m - cx) / dx_div
        ny = (Y_m - cy) / dy_div
        nx = np.clip(nx, -1.0, 1.0)
        ny = np.clip(ny, -1.0, 1.0)
        
        val_x = np.exp(-nx**2) * (1 - nx**2)**2
        val_y = np.exp(-ny**2) * (1 - ny**2)**2
        
        A = 8.0
        V = A * val_x * val_y
        Z[mask] += lambda_obs * V
        
    return Z