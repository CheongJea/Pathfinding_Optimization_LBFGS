import numpy as np
import matplotlib.pyplot as plt
import os

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


def draw_line_contour_graphs(obstacles, lambda_obs, iter,x, y, margin, potential_2D_dir, potential_3D_dir):
    X, Y = np.meshgrid(x, y)
    Z = calculate_potential_grid(X, Y, obstacles, margin, lambda_obs)


    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)

    levels = 20
    contour_lines = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', linewidths=1.5)
    

    for obs in obstacles:

        rect_x = [obs[0], obs[1], obs[1], obs[0], obs[0]]
        rect_y = [obs[2], obs[2], obs[3], obs[3], obs[2]]
        ax1.plot(rect_x, rect_y, 'r-', linewidth=2)
        

        m_x = [obs[0]-margin, obs[1]+margin, obs[1]+margin, obs[0]-margin, obs[0]-margin]
        m_y = [obs[2]-margin, obs[2]-margin, obs[3]+margin, obs[3]+margin, obs[2]-margin]
        #ax1.plot(m_x, m_y, 'k--', linewidth=1, alpha=0.5)

    ax1.set_xlabel("X",fontweight='bold', fontsize = 12)
    ax1.set_ylabel("Y",fontweight='bold', fontsize = 12)
    ax1.set_aspect('equal')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    

    cbar1 = fig1.colorbar(contour_lines, ax=ax1)
    plt.savefig(os.path.join(potential_2D_dir, f"potential_{iter}.png"))

    # Figure 2: 3D Surface Plot (참조용)
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax2.set_xlabel("X",fontweight='bold')
    ax2.set_ylabel("Y",fontweight='bold')
    ax2.set_zlabel("Cost",fontweight='bold')
    ax2.view_init(elev=40, azim=220)
    
    cbar2 = fig2.colorbar(surf, ax=ax2, shrink=0.6, aspect=15)
    plt.savefig(os.path.join(potential_3D_dir, f"potential_{iter}.png"))
