import pandas as pd
import numpy as np
import os

def Save_History2Csv(history, Save_dir, name):
    df_history = pd.DataFrame({
        "loss": history["loss"],
        "grad_norm": history["grad_norm"],
        "L_length": history["L_length"],
        "L_smooth": history["L_smooth"],
        "L_obs": history["L_obs"],
        "STD": history["STD"]
    })

    df_history.insert(0, "step", range(1, len(df_history) + 1))

    # CSV 저장
    csv_path = os.path.join(Save_dir, name)
    df_history.to_csv(csv_path, index=False)

    print(f"History saved to {csv_path}")

def History_Save(history, f, grad_norm, lambda_length, lambda_smooth, lambda_obs, L1, L2, L3, k):
    history["loss"].append(f)
    history["grad_norm"].append(grad_norm)
    history["L_length"].append(lambda_length*L1)
    history["L_smooth"].append(lambda_smooth*L2)
    history["L_obs"].append(lambda_obs*L3)

    if k > 10:
        history['STD'].append(np.std(history['loss'][-10:-1]))
    else:
        history['STD'].append(np.std(history['loss']))

    return history
