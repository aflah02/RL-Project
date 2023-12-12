import json
import numpy as np
import plotly.graph_objects as go

def plot(data: list, target_column: str):
    if target_column not in data[0] or target_column not in ("eval_accuracy"):
        raise ValueError(f"target_column {target_column} not in data")
    
    if target_column == "cumulative_estimated_reward":
        rewards = []
        for i in range(len(data)):
            rewards.append(list(data[i]["cumulative_estimated_reward"].values()))
        rewards = np.array(rewards)

        fig = go.Figure()

        for i in range(rewards.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(rewards.shape[0]),
                    y=rewards[:, i],
                    mode="lines",
                    name=f"arm {i}"
                )
            )
        fig.update_layout(
            title="Cumulative Estimated Reward",
            xaxis_title="Round",
            yaxis_title="Cumulative Estimated Reward"
        )
        fig.show()
    
    elif target_column == "eval_accuracy":
        accuracies = []
        for i in range(len(data)):
            if "eval_accuracy" in data[i]:
                accuracies.append(data[i]["eval_accuracy"])
        accuracies = np.array(accuracies)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(accuracies.shape[0]),
                y=accuracies,
                mode="lines",
                name="accuracy"
            )
        )

        fig.update_layout(
            title="Accuracy",
            xaxis_title="Round",
            yaxis_title="Accuracy"
        )

        fig.show()
