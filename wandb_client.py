import netrc
from pathlib import Path
from typing import Any, Sequence
from contextlib import suppress
import wandb
from wandb.sdk.wandb_settings import Settings


def logout_wandb():
    netrc_path = Path("/root/.netrc")
    if not netrc_path.exists():
        return

    n = netrc.netrc(netrc_path)

    if "api.wandb.ai" in n.hosts:
        del n.hosts["api.wandb.ai"]

        netrc_path.write_text(repr(n))


class WeightsAndBiasesClient:
    def __init__(
        self,
        api_key: str,
        project: str,
        config: dict,
        sample_prompts: list[str],
        entity: str | None,
        name: str | None,
    ):
        self.api_key = api_key
        self.sample_prompts = sample_prompts
        wandb.login(key=self.api_key, verify=True)
        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config,
                save_code=False,
                settings=Settings(_disable_machine_info=True),
            )
        except Exception as e:
            raise ValueError(f"Failed to log in to Weights & Biases: {e}")

    def log_loss(self, loss_dict: dict[str, Any], step: int | None):
        try:
            wandb.log(data=loss_dict, step=step)
        except Exception as e:
            print(f"Failed to log to Weights & Biases: {e}")

    def log_samples(self, image_paths: Sequence[Path], step: int | None):
        data = {}
        
        # If we have more paths than prompts, just log the paths without prompt annotations
        if len(image_paths) > len(self.sample_prompts) or not self.sample_prompts:
            for i, path in enumerate(image_paths):
                suffix = path.suffix.lower()
                if suffix in ['.mp4', '.avi', '.mov', '.webm']:
                    try:
                        data[f"video/{path.name}"] = wandb.Video(str(path))
                    except Exception as e:
                        print(f"Failed to log video {path}: {e}")
                else:
                    try:
                        data[f"image/{path.name}"] = wandb.Image(str(path))
                    except Exception as e:
                        print(f"Failed to log image {path}: {e}")
        else:
            # Map prompts to paths
            for prompt, path in zip(self.sample_prompts, image_paths):
                key = f"samples/{truncate(prompt)}"
                suffix = path.suffix.lower()
                if suffix in ['.mp4', '.avi', '.mov', '.webm']:
                    try:
                        data[key] = wandb.Video(str(path))
                    except Exception as e:
                        print(f"Failed to log video {path} for prompt {prompt}: {e}")
                else:
                    try:
                        data[key] = wandb.Image(str(path))
                    except Exception as e:
                        print(f"Failed to log image {path} for prompt {prompt}: {e}")
        
        if data:
            try:
                wandb.log(data=data, step=step)
                print(f"Logged {len(data)} samples to W&B at step {step}")
            except Exception as e:
                print(f"Failed to log to Weights & Biases: {e}")

    def save_weights(self, lora_path: Path):
        try:
            wandb.save(lora_path)
        except Exception as e:
            print(f"Failed to save to Weights & Biases: {e}")

    def finish(self):
        with suppress(Exception):
            wandb.finish()


def truncate(text, max_chars=50):
    if len(text) <= max_chars:
        return text
    half = (max_chars - 3) // 2
    return f"{text[:half]}...{text[-half:]}"
