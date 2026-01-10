from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os

def Save_Frame(x, N, start, goal, obstacles,k, frame_dir):
    path_opt = x.reshape(N, 2)
    path_full = np.vstack([start, path_opt, goal])
    fig, ax = plt.subplots(figsize=(8,7))
    plt.plot(path_full[:, 0], path_full[:, 1], 's-',color='k', label='Path')
    # for rect in obstacles:
    #     plt.plot([rect[0], rect[1], rect[1], rect[0], rect[0]],
    #              [rect[2], rect[2], rect[3], rect[3], rect[2]], 'r-')
    
    for rect in obstacles:
        x1, x2, y1, y2, _, _= rect
        width = x2 - x1
        height = y2 - y1
        ax.add_patch(Rectangle((x1, y1), width, height,facecolor='r', alpha=1, edgecolor='r'))    
    
    plt.scatter(start[0], start[1], c='red', s=100)
    plt.scatter(goal[0], goal[1], c='blue', s=100)
    plt.xlim(-5, 35)
    plt.ylim(-5, 35)
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.grid(True)
    plt.title(f"Iteration {k}")
    plt.savefig(os.path.join(frame_dir, f"frame_{k:04d}.png"))
    plt.close()



def Plot_Optimized_Path(x, N, start, goal, obstacles):
    path_opt = x.reshape(N,2)
    path_full = np.vstack([start, path_opt, goal])
    fig, ax = plt.subplots(figsize=(8,7))
    plt.plot(path_full[:,0], path_full[:,1], 's-',color='k', label='Path')
    
    
    # for rect in obstacles:
    #     plt.plot([rect[0],rect[1],rect[1],rect[0],rect[0]],
    #             [rect[2],rect[2],rect[3],rect[3],rect[2]], 'r-')
    
    # 사각형 채운 버전, 맘에 안 들면 그냥 버려야지
    for rect in obstacles:
        x1, x2, y1, y2, _, _= rect
        width = x2 - x1
        height = y2 - y1
        ax.add_patch(Rectangle((x1, y1), width, height,facecolor='r', alpha=1, edgecolor='r'))
    
    plt.scatter(start[0], start[1], c='red', s=100, label='Start')
    plt.scatter(goal[0], goal[1], c='blue', s=100, label='Goal')
    plt.grid(False)
    plt.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.show()


def Plot_Optimized_All_History(history):
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(history["L_length"], label="Path Length Loss", color ='red', linewidth=2)
    plt.plot(history["L_smooth"], label="Smoothness Loss", color= 'blue', linewidth=2)
    plt.plot(history["L_obs"], label="Obstacle Loss", color = 'teal', linewidth=2)
    plt.plot(history["loss"], label="Total Loss", linewidth=2, color = 'black')
    plt.xlabel("Iteration", fontweight='bold', fontsize = 12)
    plt.ylabel("Loss", fontweight='bold', fontsize = 12)
    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.legend(frameon=False, fontsize = 12)
    #plt.title("Optimization History")
    plt.show()


def Plot_Optimized_History(history, type, ylabel, label):


    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(history[type], color='k',label=label, linewidth=2)
    plt.xlabel("Iteration", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.legend(frameon=False, fontsize = 12)
    plt.show()



