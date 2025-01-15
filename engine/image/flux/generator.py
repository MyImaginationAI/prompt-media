import datetime
import io
import json
import logging
import os
from typing import List, Optional

import websocket
from PIL import Image

from libs.config import config

from .config_schema import WorkflowConfig
from .workflow import WorkflowManager


class ImageGenerator:
    """Handles the image generation process."""

    def __init__(self, workflow_manager: WorkflowManager, server_address: Optional[str] = None):
        """Initialize the image generator."""
        self.workflow_manager = workflow_manager
        self.server_address = server_address or f"{config.server.host}:{config.server.port}"
        self.logger = logging.getLogger(__name__)

    def _setup_websocket(self, client_id: str) -> websocket.WebSocket:
        """Set up WebSocket connection."""
        ws_url = f"ws://{self.server_address}/ws?clientId={client_id}"
        self.logger.info(f"🔌 Connecting to WebSocket: {ws_url}")
        try:
            return websocket.create_connection(ws_url)
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to WebSocket: {e}")
            raise

    def _update_workflow(self, workflow_api: dict, prompt: str, config: WorkflowConfig, seed: int) -> dict:
        """Update workflow API with current configuration."""
        try:
            # Find and update CLIP Text Encode node
            prompt_nodes = [
                k
                for k, v in workflow_api.items()
                if v.get("class_type") == "CLIPTextEncode" and "_meta" in v and "Prompt" in v["_meta"].get("title", "")
            ]
            if prompt_nodes:
                workflow_api[prompt_nodes[0]]["inputs"]["text"] = prompt

            # Find and update KSampler node
            sampler_nodes = [k for k, v in workflow_api.items() if v.get("class_type") == "KSampler"]
            if sampler_nodes:
                workflow_api[sampler_nodes[0]]["inputs"].update({"seed": seed, "steps": config.steps, "cfg": config.cfg})

            # Find and update Empty Latent Image node
            latent_nodes = [k for k, v in workflow_api.items() if v.get("class_type") == "EmptyLatentImage"]
            if latent_nodes:
                workflow_api[latent_nodes[0]]["inputs"].update({"width": config.width, "height": config.height})

            # Find and update SaveImage node
            save_nodes = [k for k, v in workflow_api.items() if v.get("class_type") == "SaveImage"]
            if save_nodes:
                current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
                workflow_api[save_nodes[0]]["inputs"]["filename_prefix"] = f"{current_date}_"

            self.logger.info(f"⚙️ Workflow updated with prompt: {prompt}")
            return workflow_api
        except Exception as e:
            self.logger.error(f"❌ Failed to update workflow: {e}")
            raise

    def _get_images(self, ws: websocket.WebSocket, workflow_api: dict) -> List[bytes]:
        """Get generated images from the API."""
        try:
            ws.send(json.dumps({"prompt": workflow_api}))
            output_images = []

            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message.get("type") == "executing":
                        self.logger.info("🎨 Generating image...")
                    elif message.get("type") == "executed":
                        break
                else:
                    output_images.append(out)

            return output_images
        except Exception as e:
            self.logger.error(f"❌ Failed to get images: {e}")
            raise

    def _save_images(self, images: List[bytes], output_dir: str, count: int, seed: int, workflow_metadata: dict) -> List[str]:
        """Save generated images to output directory."""
        # Create date-based directory structure
        current_time = datetime.datetime.now()
        date_dir = os.path.join(
            output_dir,
            current_time.strftime("%Y"),
            current_time.strftime("%m"),
            current_time.strftime("%d"),
            current_time.strftime("%H%M")
        )
        os.makedirs(date_dir, exist_ok=True)

        image_paths = []
        for idx, image_data in enumerate(images):
            try:
                image = Image.open(io.BytesIO(image_data))
                filename = f"{count + idx + 1:04d}_seed_{seed}.png"
                filepath = os.path.join(date_dir, filename)
                image.save(filepath)
                image_paths.append(filepath)
                self.logger.info(f"💾 Saved image to: {filepath}")

                # Save workflow metadata alongside the image
                metadata_path = os.path.splitext(filepath)[0] + '.json'
                with open(metadata_path, 'w') as f:
                    json.dump(workflow_metadata, f, indent=2)
                self.logger.info(f"💾 Saved workflow metadata to: {metadata_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save image {idx + 1}: {e}")
                continue

        return image_paths

    def generate(self, prompt: str, workflow_path: str, output_dir: str, count: int, seed: int, client_id: str) -> List[str]:
        """Generate images based on the given prompt and configuration."""
        try:
            # Get workflow configuration
            workflow_type, config = self.workflow_manager.get_workflow_config(workflow_path)

            # Load workflow API
            with open(workflow_path, "r") as f:
                workflow_api = json.load(f)

            # Setup WebSocket connection
            ws = self._setup_websocket(client_id)

            # Update workflow with current settings
            workflow_api = self._update_workflow(workflow_api, prompt, config, seed)

            # Generate workflow and get metadata
            _, workflow_metadata = self.workflow_manager.generate_workflow(workflow_path, output_dir, prompt)

            # Generate and get images
            images = self._get_images(ws, workflow_api)
            if not images:
                self.logger.error("❌ No images were returned from the API")
                return []

            # Save images with workflow metadata
            return self._save_images(images, output_dir, count, seed, workflow_metadata)

        except Exception as e:
            self.logger.error(f"❌ Error occurred while generating image: {e}")
            raise
        finally:
            if "ws" in locals():
                ws.close()
