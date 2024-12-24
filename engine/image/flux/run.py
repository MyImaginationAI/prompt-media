import argparse
import datetime
import io
import json
import logging
import os
import platform
import random
import shutil
import uuid
from itertools import product
from pathlib import Path
from urllib import parse, request

import websocket
import yaml
from jsonschema import ValidationError, validate
from PIL import Image

from libs.config import config

# ANSI escape sequences for colored output - handle Windows separately


class TextColors:
    def __init__(self):
        self.use_colors = platform.system() != "Windows" or os.environ.get("FORCE_COLOR")
        self._setup_colors()

    def _setup_colors(self):
        if self.use_colors:
            self.HEADER = "\033[95m"
            self.OKBLUE = "\033[94m"
            self.OKGREEN = "\033[92m"
            self.WARNING = "\033[93m"
            self.FAIL = "\033[91m"
            self.ENDC = "\033[0m"
            self.BOLD = "\033[1m"
            self.UNDERLINE = "\033[4m"
            self.HIGHLIGHT = "\033[96m"
        else:
            self.HEADER = ""
            self.OKBLUE = ""
            self.OKGREEN = ""
            self.WARNING = ""
            self.FAIL = ""
            self.ENDC = ""
            self.BOLD = ""
            self.UNDERLINE = ""
            self.HIGHLIGHT = ""


# Initialize colors globally
COLORS = TextColors()

# Simple text symbols for different message types (no emojis)


class Symbols:
    INFO = "[i]"
    SUCCESS = "[+]"
    WARNING = "[!]"
    ERROR = "[x]"
    PROCESSING = "[*]"
    IMAGE = "[>]"
    SAVE = "[s]"
    TIME = "[t]"
    CONFIG = "[c]"


# Choose between emojis and simple symbols based on platform
if platform.system() != "Windows" or os.environ.get("FORCE_UNICODE"):

    class Emojis:
        INFO = "ℹ️ "
        SUCCESS = "✅ "
        WARNING = "⚠️ "
        ERROR = "❌ "
        PROCESSING = "⚙️ "
        IMAGE = "🖼️ "
        SAVE = "💾 "
        TIME = "⏱️ "
        CONFIG = "⚙️ "

    MessageSymbols = Emojis
else:
    MessageSymbols = Symbols

# Schema definition for YAML validation
FLUX_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["prompt_settings", "prompts"],
    "properties": {
        "prompt_settings": {
            "type": "object",
            "properties": {
                "prefix": {"type": "string"},
                "negative": {"type": "string"}
            },
            "required": ["prefix", "negative"]
        },
        "prompts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        },
        "variations": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "string"}},
                    "type": {"type": "string", "enum": ["static", "sequential", "random"]}
                },
                "required": ["values", "type"]
            }
        },
        "workflows": {
            "type": "object",
            "properties": {
                "dev": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1},
                        "cfg": {"type": "number", "minimum": 1},
                        "width": {"type": "integer", "minimum": 64},
                        "height": {"type": "integer", "minimum": 64},
                        "seeds": {"type": "array", "items": {"type": "integer"}}
                    }
                },
                "schnell": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "minimum": 1},
                        "cfg": {"type": "number", "minimum": 1}
                    }
                }
            }
        },
        "loras": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "strength_model": {"type": "number"},
                "strength_clip": {"type": "number"}
            }
        }
    }
}


def validate_config(config):
    """Validate the configuration against the schema."""
    try:
        validate(instance=config, schema=FLUX_CONFIG_SCHEMA)
        logging.info(f"{MessageSymbols.SUCCESS} Configuration validation passed")
        return True
    except ValidationError as e:
        logging.error(f"{MessageSymbols.ERROR} Configuration validation failed:")
        logging.error(f"{MessageSymbols.ERROR} {e.message}")
        logging.error(f"{MessageSymbols.ERROR} Path: {' -> '.join(str(p) for p in e.path)}")
        return False


def setup_logging(output_dir: Path):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'flux.log')
        ]
    )


def read_file(file_path):
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as file:
        if file_path.suffix.lower() in [".json"]:
            data = json.load(file)
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(file)
        else:
            raise ValueError(f"File format not supported: {file_path}")
    return data


def remove_duplicates(prompt):
    words = prompt.split(", ")
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    return ", ".join(unique_words)


def get_image(filename, subfolder, folder_type, server_address):
    """Get image data from the ComfyUI server."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = parse.urlencode(data)
    url = f"http://{server_address}/view?{url_values}"
    try:
        with request.urlopen(url) as response:
            return response.read()
    except Exception as e:
        raise RuntimeError(f"Failed to get image from server: {str(e)}")


def get_history(prompt_id, server_address):
    with request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def check_server_status(server_address):
    """Check if the ComfyUI server is responsive"""
    try:
        response = request.urlopen(f"http://{server_address}/system_stats")
        if response.status == 200:
            stats = json.loads(response.read())
            logging.info(f"{MessageSymbols.INFO} Server is running - Memory: {stats.get('ram', {}).get('free', 'N/A')}MB free")
            return True
    except Exception as e:
        logging.error(f"{MessageSymbols.ERROR} Server health check failed: {str(e)}")
        return False


def create_websocket_connection(server_address, client_id, timeout=30):
    """Create a WebSocket connection with proper error handling"""
    websocket.enableTrace(True)  # Enable tracing for all connections
    websocket.setdefaulttimeout(timeout)

    # Ensure we have the port in the address
    if ":" not in server_address:
        server_address += ":8188"
    ws_url = f"ws://{server_address}/ws?clientId={client_id}"

    try:
        logging.info(f"{MessageSymbols.INFO} Attempting connection: {ws_url}")
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        logging.info(f"{MessageSymbols.SUCCESS} WebSocket connection established")
        return ws
    except Exception as err:
        raise ConnectionError(f"Failed to connect to WebSocket: {str(err)}")


def get_images(ws, workflow_api, server_address, client_id):
    """Get generated images from the ComfyUI server"""
    try:
        # Verify server is responsive
        if not check_server_status(server_address):
            raise ConnectionError("ComfyUI server is not responding")

        # Submit prompt
        p = {"prompt": workflow_api, "client_id": client_id}
        data = json.dumps(p).encode("utf-8")
        
        # Debug log the workflow being sent
        logging.debug(f"Submitting workflow: {json.dumps(workflow_api, indent=2)}")

        req = request.Request(f"http://{server_address}/prompt", data=data)
        req.add_header("Content-Type", "application/json")

        try:
            with request.urlopen(req) as response:
                response_data = json.loads(response.read())
                prompt_id = response_data["prompt_id"]
                logging.info(f"{MessageSymbols.INFO} Prompt submitted successfully. ID: {prompt_id}")
        except Exception as e:
            raise ConnectionError(f"Failed to submit prompt to server: {str(e)}")

        # Wait for execution
        output_images = {}
        timeout_counter = 0
        max_timeout = 600  # 10 minutes maximum wait time
        retry_interval = 5  # Check every 5 seconds

        while True:
            try:
                ws.settimeout(retry_interval)
                out = ws.recv()
                timeout_counter = 0  # Reset counter on successful receive

                if isinstance(out, str):
                    message = json.loads(out)
                    logging.debug(f"Received message type: {message['type']}")

                    if message["type"] == "executing":
                        data = message["data"]
                        if data.get("node", None) is None and data.get("prompt_id") == prompt_id:
                            logging.info(f"{MessageSymbols.SUCCESS} Generation completed")
                            break
                        # Log progress if available
                        if "value" in data and "max" in data:
                            progress = (data["value"] / data["max"]) * 100
                            logging.info(f"{MessageSymbols.PROCESSING} Generation progress: {progress:.1f}%")
                    elif message["type"] == "error":
                        error_msg = message.get("data", {}).get("error", "Unknown error occurred")
                        raise RuntimeError(f"Server error: {error_msg}")
                    elif message["type"] == "status":
                        status = message.get("data", {}).get("status", {})
                        if status:
                            logging.info(f"{MessageSymbols.INFO} Server status: {status}")
                else:
                    logging.debug("Received binary data (preview)")

            except websocket.WebSocketTimeoutException:
                timeout_counter += retry_interval
                if timeout_counter >= max_timeout:
                    raise TimeoutError(f"Image generation timed out after {max_timeout} seconds")

                # Check server status periodically
                if timeout_counter % 30 == 0:  # Every 30 seconds
                    if not check_server_status(server_address):
                        raise ConnectionError("Lost connection to ComfyUI server")

                logging.warning(f"{MessageSymbols.WARNING} Waiting for response... ({timeout_counter} seconds)")
                continue

            except Exception as e:
                raise RuntimeError(f"WebSocket error: {str(e)}")

        # Get generated images
        try:
            history = get_history(prompt_id, server_address)[prompt_id]
            for node_id in history["outputs"]:
                node_output = history["outputs"][node_id]
                if "images" in node_output:
                    images_output = []
                    for image in node_output["images"]:
                        image_data = get_image(image["filename"], image["subfolder"], image["type"], server_address)
                        images_output.append(image_data)
                    output_images[node_id] = images_output
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve images: {str(e)}")

        return output_images

    except Exception as e:
        ws.close()
        raise e


def get_datetime_output_path(base_dir: Path, current_time: datetime.datetime) -> Path:
    """Generate a datetime-based directory path.

    Args:
        base_dir: Base directory path
        current_time: Current datetime object

    Returns:
        Path object with the full datetime-based directory structure
    """
    # Format: YYYY/mm/dd/HHMM
    date_path = current_time.strftime("%Y/%m/%d")
    time_path = current_time.strftime("%H%M")
    return base_dir / date_path / time_path


def save_images(images, output_dir, global_count, seed, current_time, prompt_media_path=None):
    """Save generated images to the output directory with proper organization."""
    # Convert the output_dir to Path object if it's a string
    output_base = Path(output_dir)

    # Use the current time to generate the datetime-based directory structure
    output_path = get_datetime_output_path(output_base, current_time)

    # Create the media and images directories
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy the prompt media file if provided
    if prompt_media_path:
        prompt_media_path = Path(prompt_media_path)
        if prompt_media_path.exists():
            dest_prompt_path = output_path / prompt_media_path.name
            shutil.copy2(prompt_media_path, dest_prompt_path)
            logging.info(f"{MessageSymbols.SAVE} Copied prompt file to: {dest_prompt_path.relative_to(output_base)}")
            print(f"{COLORS.OKGREEN}{MessageSymbols.SAVE} Copied prompt file to: {dest_prompt_path.relative_to(output_base)}{COLORS.ENDC}")

    saved_paths = []
    image_count = 0

    # Iterate through each node's image list
    for node_id, image_list in images.items():
        for image_data in image_list:
            # Format: {count:03d}_seed_{seed}.png (e.g., 001_seed_12345.png)
            filename = f"{global_count + image_count:03d}_seed_{seed}.png"
            image_path = output_path / filename

            # Save the image
            image = Image.open(io.BytesIO(image_data))
            image.save(str(image_path))
            saved_paths.append(image_path)

            # Log the saved image
            logging.info(f"{MessageSymbols.SAVE} Saved image to: {image_path.relative_to(output_base)}")
            print(f"{COLORS.OKGREEN}{MessageSymbols.SAVE} Saved image to: {image_path.relative_to(output_base)}{COLORS.ENDC}")

            image_count += 1

    return saved_paths


def is_visual_element(element):
    non_visual_keywords = ["abstract", "concept", "idea", "feeling", "emotion", "success", "milestone"]
    return not any(keyword in element.lower() for keyword in non_visual_keywords)


def generate_prompts(prefix_prompt, base_prompt, suffix_files_types, user_preference=None, sequential_state=None):
    """
    Generate prompts with OOM variations.

    Args:
        prefix_prompt (str): Prefix for all prompts
        base_prompt (str): Base prompt to build upon
        suffix_files_types (list): List of tuples (key, value, type) for OOM variations
        user_preference (str, optional): User preference for filtering options
        sequential_state (dict, optional): Current state of sequential OOMs

    Returns:
        tuple: (list of prompts, updated sequential state)
    """
    logging.debug(f"{MessageSymbols.PROCESSING} Generating prompts with base: {base_prompt}")

    # Initialize prompt with base content
    combined_prompt = f"{prefix_prompt} {base_prompt}"

    if user_preference:
        logging.info(f"{MessageSymbols.INFO} Applying user preference filter: {user_preference}")

    # Split variations by type
    sequential_variations = []
    random_variations = []
    static_variations = []

    for key, value, suffix_type in suffix_files_types:
        values = value.split(", ")
        if user_preference:
            values = [v for v in values if user_preference.lower() in v.lower()]

        if suffix_type == "sequential":
            sequential_variations.append(values)
        elif suffix_type == "random":
            random_variations.append(values)
        elif suffix_type == "static":
            static_variations.append(value)

    # Generate all combinations of sequential variations
    sequential_combinations = list(product(*sequential_variations)) if sequential_variations else [()]

    # Generate prompts for each sequential combination
    generated_prompts = []
    for seq_combo in sequential_combinations:
        suffix_parts = list(seq_combo) + static_variations

        # Handle random variations
        random_parts = []
        for random_values in random_variations:
            if random_values:  # Only add if there are values available
                selected = random.choice(random_values)
                random_parts.append(selected)

        suffix_parts.extend(random_parts)

        # Combine all parts
        final_prompt = f"{combined_prompt}, {', '.join(suffix_parts)}"
        generated_prompts.append(remove_duplicates(final_prompt))

    return generated_prompts, sequential_state


def parse_suffix_files(variations):
    """Parse variations from the config file."""
    suffix_files_types = []
    if not variations:
        return suffix_files_types

    for key, variation in variations.items():
        # Convert list values to comma-separated string
        values = ", ".join(variation.get("values", []))
        variation_type = variation.get("type", "")
        suffix_files_types.append((key, values, variation_type))

    return suffix_files_types


def get_vram_stats(server_address):
    """Get VRAM statistics from the ComfyUI server, handling WSL case"""
    try:
        response = request.urlopen(f"http://{server_address}/vram")
        data = json.loads(response.read().decode("utf-8"))

        # Convert bytes to MB for better readability
        vram_free = data.get("vram_free", 0)
        total_vram = data.get("total_vram", 0)

        vram_free_mb = vram_free / (1024 * 1024)  # Convert to MB
        total_vram_mb = total_vram / (1024 * 1024)  # Convert to MB

        # Calculate usage percentage
        vram_usage_percent = ((total_vram - vram_free) / total_vram) * 100 if total_vram > 0 else 0

        logging.info(
            f"{MessageSymbols.INFO} VRAM Stats - Free: {vram_free_mb:.2f}MB, Total: {total_vram_mb:.2f}MB, Usage: {vram_usage_percent:.1f}%"
        )
        return vram_free, total_vram, vram_usage_percent

    except Exception as e:
        logging.warning(f"{MessageSymbols.WARNING} Failed to get VRAM stats: {str(e)}")
        return None, None, None


def load_workflow(workflow_path, prompt_data=None, prompt=None, negative_prompt=None):
    """
    Load and customize workflow based on prompt data.
    
    Args:
        workflow_path (str): Path to the workflow JSON file
        prompt_data (dict): The loaded prompt data containing lora information
        prompt (str): The prompt to use in the workflow
        negative_prompt (str): The negative prompt to use in the workflow
        
    Returns:
        dict: The loaded and customized workflow
    """
    from engine.image.flux.template_processor import Jinja2TemplateProcessor, WorkflowContextBuilder
    
    # Check if workflow requires LoRA by looking for LoraLoader
    with open(workflow_path, 'r') as f:
        workflow_content = f.read()
        requires_lora = 'LoraLoader' in workflow_content
    
    # Build context from prompt data
    workflow_config = {
        "filename_prefix": "output",
        "negative_text": negative_prompt if negative_prompt else "",
        "steps": prompt_data.get("workflows", {}).get("dev", {}).get("steps", prompt_data.get("steps", 25)) if prompt_data else 25,
        "cfg": prompt_data.get("workflows", {}).get("dev", {}).get("cfg", prompt_data.get("cfg", 1.0)) if prompt_data else 1.0,
        "width": prompt_data.get("workflows", {}).get("dev", {}).get("width", prompt_data.get("width", 768)) if prompt_data else 768,
        "height": prompt_data.get("workflows", {}).get("dev", {}).get("height", prompt_data.get("height", 768)) if prompt_data else 768,
        "seeds": prompt_data.get("workflows", {}).get("dev", {}).get("seeds", prompt_data.get("seeds", [1])) if prompt_data else [1]
    }
    
    # Update workflow config with LoRA info if present
    lora_config = None
    if prompt_data and requires_lora:
        lora_config = prompt_data.get("loras", {}).get("lora", [{}])[0]  # Get first LoRA config
        if lora_config and "name" in lora_config:
            # Ensure the LoRA name has .safetensors extension
            if not lora_config["name"].endswith(".safetensors"):
                lora_config["name"] = f"{lora_config['name']}.safetensors"
            
            # Set strength values if not provided
            if "strength" in lora_config:
                lora_config["strength_model"] = lora_config["strength"]
                lora_config["strength_clip"] = lora_config["strength"]
            
            logging.debug(f"Using LoRA configuration: {lora_config}")
    
    # Build template context
    context = WorkflowContextBuilder.build_context(
        prompt=prompt,
        workflow_config=workflow_config,
        lora_config=lora_config,
        requires_lora=requires_lora
    )
    
    # Process template
    template_processor = Jinja2TemplateProcessor()
    try:
        workflow_content = template_processor.process_template(workflow_path, context)
        workflow = json.loads(workflow_content)
        logging.debug(f"Generated workflow: {json.dumps(workflow, indent=2)}")
        return workflow
    except Exception as e:
        logging.error(f"Error processing workflow template: {e}")
        raise


def select_workflow(prompt_data, workflow_path=None):
    """
    Select the appropriate workflow based on prompt configuration.
    
    Args:
        prompt_data (dict): The loaded prompt data
        workflow_path (str, optional): User-specified workflow path
        
    Returns:
        str: Path to the selected workflow JSON file
    """
    base_dir = Path(__file__).parent / "workflows" / "api"
    
    # Handle user-specified workflow
    if workflow_path:
        if workflow_path == "schnell":
            return str(base_dir / "flux1-schnell-fp8-api.json")
        return workflow_path
    
    # Default workflow
    return str(base_dir / "flux1-dev-fp8-api.json")


def generate_image(
    prompt: str,
    negative_prompt: str,
    global_count: int,
    seed: int,
    steps: int,
    width: int,
    height: int,
    cfg: float,
    output_dir: str,
    server_address: str,
    client_id: str,
    workflow_path: str,
    current_time: datetime.datetime,
    prompt_media_path: str | None = None,
) -> None:
    try:
        # Check server status
        if not check_server_status(server_address):
            raise ConnectionError(f"ComfyUI server at {server_address} is not responsive")

        # Load prompt data if path provided
        prompt_data = read_file(prompt_media_path) if prompt_media_path else None

        # Load and customize workflow
        workflow = load_workflow(workflow_path, prompt_data=prompt_data, prompt=prompt, negative_prompt=negative_prompt)
        if not workflow:
            raise ValueError("Failed to load workflow")

        # Save workflow metadata as JSON
        workflow_metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "width": width,
            "height": height,
            "cfg": cfg,
            "seed": seed,
            "workflow_path": workflow_path,
            "prompt_media_path": prompt_media_path,
            "prompt_data": prompt_data,
            "workflow": workflow,
            "timestamp": current_time.isoformat()
        }
        
        # Generate output path for metadata
        output_dir_path = Path(output_dir)
        metadata_path = output_dir_path / f"{global_count:03d}_seed_{seed}.json"
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(workflow_metadata, f, indent=2)
        logging.info(f"{MessageSymbols.SAVE} Saved workflow metadata to: {metadata_path}")

        # Create WebSocket connection
        ws = create_websocket_connection(server_address, client_id)
        if not ws:
            raise ConnectionError("Failed to create WebSocket connection")

        # Get generated images
        images = get_images(ws, workflow, server_address, client_id)
        if not images:
            raise ValueError("No images were generated")

        # Save the generated images
        save_images(images, output_dir, global_count, seed, current_time, prompt_media_path)

    except Exception as e:
        logging.error(f"{MessageSymbols.ERROR} Failed to generate image: {e}")
        raise


def main(
    workflow_path: str,
    prompt_media_path: str | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
    user_preference: str | None = None,
    server_address: str | None = None,
    client_id: str | None = None,
) -> str | None:
    """Main function to run the image generation process."""
    try:
        # Load and validate prompt data
        if not prompt_media_path:
            raise ValueError("prompt_media_path must be provided")
        
        prompt_data = read_file(prompt_media_path)
        if not prompt_data:
            raise ValueError(f"Failed to load prompt data from {prompt_media_path}")

        # Select appropriate workflow based on prompt data
        workflow_path = select_workflow(prompt_data, workflow_path)
        if not workflow_path:
            raise ValueError("Failed to select a workflow")

        if dry_run:
            logging.info(f"{MessageSymbols.SUCCESS} Dry run complete. Generated prompts saved to output/generated_prompts.json")
            return workflow_path

        # Handle None values for optional parameters
        output_dir = output_dir or "output"
        server_address = server_address or "http://127.0.0.1:8188"
        client_id = client_id or str(uuid.uuid4())

        # Convert string paths to Path objects
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging with Path object
        setup_logging(output_dir_path)
        logger = logging.getLogger(__name__)

        # Validate configuration
        if not validate_config(prompt_data):
            raise ValueError("Invalid configuration")

        negative_prompt = prompt_data.get("prompt_settings", {}).get("negative", "")
        prefix_prompt = prompt_data.get("prompt_settings", {}).get("prefix", "")
        prompts = prompt_data.get("prompts", [])
        variations = prompt_data.get("variations", {})

        suffix_files_types = parse_suffix_files(variations)
        sequential_state = None  # Initialize sequential state

        # Get seeds from prompt data or generate random seed
        seeds = prompt_data.get("workflows", {}).get("dev", {}).get("seeds", None)
        if not seeds:  # If seeds is None or empty list
            random_seed = random.randint(1, 1000000)  # Generate random seed between 1 and 1,000,000
            seeds = [random_seed]  # Use random seed
            logging.info(f"{MessageSymbols.INFO} No seeds specified, using random seed: {random_seed}")

        count = 0
        current_time = datetime.datetime.now()
        for seed in seeds:
            for base_prompt in prompts:
                generated_prompts, sequential_state = generate_prompts(
                    prefix_prompt, base_prompt, suffix_files_types, user_preference, sequential_state
                )
                for generated_prompt in generated_prompts:
                    logging.info(f"{MessageSymbols.PROCESSING} Executing Prompt: {generated_prompt}")
                    print(f"{COLORS.HIGHLIGHT} Executing Prompt: {generated_prompt}{COLORS.ENDC}")
                    generate_image(
                        generated_prompt,
                        negative_prompt,
                        count,
                        seed,
                        prompt_data.get("workflows", {}).get("dev", {}).get("steps", prompt_data.get("steps", 20)),
                        prompt_data.get("workflows", {}).get("dev", {}).get("width", prompt_data.get("width", 512)),
                        prompt_data.get("workflows", {}).get("dev", {}).get("height", prompt_data.get("height", 512)),
                        prompt_data.get("workflows", {}).get("dev", {}).get("cfg", prompt_data.get("cfg", 1.0)),
                        output_dir,
                        server_address,
                        client_id,
                        workflow_path,
                        current_time,
                        prompt_media_path,
                    )
                    count += 1

        logging.info(f"{MessageSymbols.SUCCESS} Image generation complete.")
        print(f"{COLORS.OKGREEN}{MessageSymbols.SUCCESS} Image generation complete.{COLORS.ENDC}")

    except Exception as e:
        logging.error(f"{MessageSymbols.ERROR} Failed to generate images: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from JSON or YAML prompts.")
    parser.add_argument("--workflow", required=True, help="Path to workflow JSON file")
    parser.add_argument("--prompt-media", required=True, help="Path to prompt media YAML file")
    parser.add_argument("--output-dir", help="Output directory for generated images")
    parser.add_argument("--dry-run", action="store_true", help="Print workflow without executing")
    parser.add_argument("--user-preference", help="User preference for filtering options")
    parser.add_argument("--server-address", help="ComfyUI server address")
    parser.add_argument("--client-id", help="Client ID for WebSocket connection")

    args = parser.parse_args()
    main(
        workflow_path=args.workflow,
        prompt_media_path=args.prompt_media,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        user_preference=args.user_preference,
        server_address=args.server_address,
        client_id=args.client_id,
    )