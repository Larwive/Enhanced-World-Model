import gradio as gr
import subprocess
import gymnasium as gym
from gymnasium.envs.registration import registry
import os
import sys
import signal # Added for process killing

def get_all_gym_envs():
    """Returns a list of all registered gym environments."""
    all_envs = list(registry.keys())
    return [env for env in all_envs if not env.startswith('_')]

# Stores the currently running subprocess
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
    save_path,
    load_path_files,
    render_every
):
    """
    Launches the training script as a subprocess and streams its stdout/stderr
    to the Gradio interface.
    """
    
    if current_process["process"] and current_process["process"].poll() is None:
        yield "A training process is already running. Please cancel it first."
        return

    # Handle the load_path argument from the FileExplorer
    # It will now always be a list.
    load_path_arg = "" # Default from main.py
    if load_path_files and isinstance(load_path_files, list) and len(load_path_files) > 0:
        load_path_arg = load_path_files[0] # Get the first selected file
    
    cmd = [
        sys.executable,
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
        "--render-every",
        str(render_every),
        "--save-path",
        save_path,
        "--load-path",
        load_path_arg # Use the processed path
    ]
    
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": 'utf-8',
        "bufsize": 1,
    }

    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid
    
    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError:
        yield f"Error: Script 'src/main.py' not found. Make sure it's in the correct directory."
        return
    except Exception as e:
        yield f"Error launching process: {e}"
        return

    current_process["process"] = process
    yield "Training launched! Monitoring output..."
    
    output_log = []

    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                output_log.append(line)
                yield "\n".join(output_log[-50:])
        process.stdout.close()
    
    return_code = process.wait()
    current_process["process"] = None
    
    if return_code == 0:
        output_log.append("--- Training finished successfully. ---")
    elif return_code == -9 or return_code == -15:
        output_log.append("--- Training was terminated by user. ---")
    else:
        output_log.append(f"--- Training failed/stopped with code {return_code}. ---")
    
    yield "\n".join(output_log[-50:])


def cancel_training():
    """
    Sends a terminate signal to the entire process group,
    killing the main script and any child processes (like Pygame).
    """
    if current_process["process"]:
        process = current_process["process"]
        if process.poll() is None:
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                        check=True, capture_output=True
                    )
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                
                return "Stop signal sent to process group (including Pygame window)."
            except Exception as e:
                process.kill()
                return f"Error killing group, fell back to process.kill(). Error: {e}"
        else:
            current_process["process"] = None
            return "Process already finished."
    return "No training process to stop!"


def update_file_explorer_root(new_root_path):
    """Updates the FileExplorer's root directory and clears its value."""
    if not os.path.isdir(new_root_path):
        # --- FIX 2 ---
        # Explicitly set value to an empty list to satisfy the validator
        return gr.FileExplorer.update(
            info=f"Path '{new_root_path}' does not exist.", 
            value=[] 
        )
    # --- FIX 2 ---
    # Explicitly set value to an empty list here too
    return gr.FileExplorer.update(root_dir=new_root_path, value=[])


with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="blue")) as demo:
    gr.Markdown("# World Model Training Interface")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## ⚙️ Hyperparameters")
            env_name = gr.Dropdown(
                choices=get_all_gym_envs(), label="Environment Name", value="CarRacing-v3"
            )
            random_seed = gr.Number(label="Random Seed", value=42, precision=0)
            max_epoch = gr.Number(label="Max Epochs", value=200, precision=0)
            patience = gr.Number(label="Patience", value=5, precision=0)
            render_every = gr.Number(label="Display Each", value=5, precision=0)
            batch_size = gr.Number(label="Batch Size", value=64, precision=0)
            learning_rate = gr.Number(label="Learning Rate", value=1e-3)
            dropout = gr.Slider(0, 1, label="Dropout", value=0.05, step=0.05)
            
            gr.Markdown("## 💾 Paths & Rendering")
            save_path = gr.Textbox(
                label="Save/Load Root Path", 
                value="./saved_models/"
            )
            
            load_path_explorer = gr.FileExplorer(
                label="Select Model to Load (filters for .pt)",
                root_dir="./saved_models/",
                # --- FIX 1 ---
                # file_count="single",  <-- REMOVED THIS
                # The value will now always be a list, satisfying the validator.
                glob="*.pt"
            )
            
            render_mode = gr.Checkbox(label="Enable Graphics (slower)", value=True)
            
            with gr.Row():
                launch_button = gr.Button("🚀 Launch Training", variant="primary")
                cancel_button = gr.Button("🛑 Cancel Training", variant="stop")

        with gr.Column(scale=3):
            gr.Markdown("## 📋 Status & Log")
            output = gr.Textbox(
                label="Log Output", 
                lines=30, 
                max_lines=30, 
                autoscroll=True, 
                placeholder="Training log will appear here..."
            )

    save_path.submit(
        fn=update_file_explorer_root,
        inputs=[save_path],
        outputs=[load_path_explorer]
    )

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
            save_path,
            load_path_explorer,
            render_every
        ],
        outputs=output,
    )
    
    cancel_button.click(fn=cancel_training, outputs=output)

if __name__ == "__main__":
    demo.launch()