
import gradio as gr
import subprocess
import gymnasium as gym
from gymnasium.envs.registration import registry

def get_all_gym_envs():
    """Returns a list of all registered gym environments."""
    return list(registry.keys())

current_process = {"process": None}

def launch_training(
    env_name,
    random_seed,
    max_epoch,
    patience,
    batch_size,
    learning_rate,
    dropout,
    render_mode,
):
    """Launches the training script with the given parameters."""
    cmd = [
        "python",
        "src/main.py",
        "--env-name",
        env_name,
        "--random-seed",
        str(random_seed),
        "--max-epoch",
        str(max_epoch),
        "--patience",
        str(patience),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--dropout",
        str(dropout),
        "--render-mode",
        "human" if render_mode else "rgb_array",
    ]
    current_process["process"]=subprocess.Popen(cmd)
    return "Training launched!"

def cancel_training():
    if current_process["process"]:
        current_process["process"].terminate()
        current_process["process"] = None
        return "Training stopped!"
    return "No training process to stop!"

with gr.Blocks() as demo:
    gr.Markdown("# World Model Training Interface")

    with gr.Row():
        with gr.Column():
            env_name = gr.Dropdown(
                choices=get_all_gym_envs(), label="Environment Name", value="CarRacing-v2"
            )
            random_seed = gr.Number(label="Random Seed", value=42)
            max_epoch = gr.Number(label="Max Epochs", value=200)
            patience = gr.Number(label="Patience", value=5)
            batch_size = gr.Number(label="Batch Size", value=64)
            learning_rate = gr.Number(label="Learning Rate", value=1e-3)
            dropout = gr.Slider(0, 1, label="Dropout", value=0.2)
            render_mode = gr.Checkbox(label="Enable Graphics", value=True)
            launch_button = gr.Button("Launch Training")
            cancel_button = gr.Button("Cancel Training")

    output = gr.Textbox(label="Status")

    launch_button.click(
        fn=launch_training,
        inputs=[
            env_name,
            random_seed,
            max_epoch,
            patience,
            batch_size,
            learning_rate,
            dropout,
            render_mode,
        ],
        outputs=output,
    )
    
    cancel_button.click(fn=cancel_training, outputs=output)

if __name__ == "__main__":
    demo.launch()
