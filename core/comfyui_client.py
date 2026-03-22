"""
ComfyUI Client — clean API client for ComfyUI server.

Handles:
- Submitting workflows to the /prompt endpoint
- Waiting for generation to complete via WebSocket
- Retrieving generated images
"""
import json
import uuid
import asyncio
import logging
from typing import List, Optional, Callable

import aiohttp

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for the ComfyUI API."""

    def __init__(self, config: dict):
        self.base_url = config.get("url", "http://127.0.0.1:8188")
        self.client_id = config.get("client_id", str(uuid.uuid4())[:8])
        logger.info(f"ComfyUI client: {self.base_url} (client_id={self.client_id})")

    async def is_alive(self) -> bool:
        """Check if ComfyUI server is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}

        # Debug: dump workflow for troubleshooting
        import tempfile, os
        debug_path = os.path.join(tempfile.gettempdir(), "comfybot_last_workflow.json")
        try:
            with open(debug_path, "w") as f:
                json.dump(workflow, f, indent=2)
            logger.debug(f"Workflow dumped to {debug_path}")
        except Exception:
            pass

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    # Parse structured error for better logging
                    try:
                        err_data = json.loads(error_text)
                        if "error" in err_data:
                            logger.error(f"ComfyUI error: {err_data['error'].get('message', 'unknown')}")
                        if "node_errors" in err_data:
                            for nid, nerr in err_data["node_errors"].items():
                                logger.error(f"  Node {nid}: {nerr}")
                    except Exception:
                        pass
                    raise RuntimeError(f"ComfyUI rejected workflow (HTTP {resp.status}): {error_text}")
                
                data = await resp.json()
                prompt_id = data["prompt_id"]
                logger.info(f"Workflow queued: {prompt_id}")
                return prompt_id

    async def wait_for_result(
        self,
        prompt_id: str,
        timeout: int = 300,
        progress_callback: Optional[Callable] = None,
    ) -> List[bytes]:
        """Wait for workflow completion via WebSocket and return images."""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws?clientId={self.client_id}"

        import websockets

        try:
            async with websockets.connect(ws_url) as ws:
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        raise RuntimeError(f"Generation timed out after {timeout}s")

                    # Binary data = preview image, skip
                    if isinstance(raw, bytes):
                        continue

                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "progress":
                        progress = msg.get("data", {})
                        if progress_callback and progress.get("prompt_id") == prompt_id:
                            step = progress.get("value", 0)
                            total = progress.get("max", 1)
                            await progress_callback(step, total)

                    elif msg_type == "executing":
                        exec_data = msg.get("data", {})
                        if exec_data.get("prompt_id") == prompt_id and exec_data.get("node") is None:
                            # Generation complete
                            break

                    elif msg_type == "execution_error":
                        err = msg.get("data", {})
                        if err.get("prompt_id") == prompt_id:
                            raise RuntimeError(f"ComfyUI execution error: {json.dumps(err)[:500]}")

        except ImportError:
            logger.error("websockets package not installed!")
            raise
        except Exception as e:
            if "execution error" in str(e).lower() or "timed out" in str(e).lower():
                raise
            logger.warning(f"WebSocket error, falling back to polling: {e}")
            await self._poll_for_completion(prompt_id, timeout)

        # Fetch the generated images
        return await self._get_images(prompt_id)

    async def _poll_for_completion(self, prompt_id: str, timeout: int = 300):
        """Poll the /history endpoint until the prompt is done."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if prompt_id in data:
                            return
            await asyncio.sleep(2)
        raise RuntimeError(f"Generation timed out after {timeout}s")

    async def _get_images(self, prompt_id: str) -> List[bytes]:
        """Fetch generated images from ComfyUI history."""
        images = []
        async with aiohttp.ClientSession() as session:
            # Get history
            async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Failed to get history for {prompt_id}")
                history = await resp.json()

            if prompt_id not in history:
                raise RuntimeError(f"Prompt {prompt_id} not found in history")

            outputs = history[prompt_id].get("outputs", {})
            
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img_info in node_output["images"]:
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        url = f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                        async with session.get(url) as img_resp:
                            if img_resp.status == 200:
                                img_data = await img_resp.read()
                                images.append(img_data)
                                logger.info(f"Retrieved image: {filename} ({len(img_data)} bytes)")

        if not images:
            logger.warning(f"No images found in output for {prompt_id}")

        return images

    async def upload_image(self, file_path: str, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload a local image file to ComfyUI. Returns the filename on the server."""
        import os
        filename = os.path.basename(file_path)

        data = aiohttp.FormData()
        data.add_field("image", open(file_path, "rb"), filename=filename, content_type="image/png")
        if subfolder:
            data.add_field("subfolder", subfolder)
        if overwrite:
            data.add_field("overwrite", "true")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/upload/image",
                data=data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Image upload failed (HTTP {resp.status}): {error_text}")
                result = await resp.json()
                uploaded_name = result.get("name", filename)
                logger.info(f"Uploaded image: {uploaded_name}")
                return uploaded_name

    async def upload_image_bytes(self, image_data: bytes, filename: str = "upload.png",
                                  subfolder: str = "", overwrite: bool = True) -> str:
        """Upload raw image bytes to ComfyUI. Returns the filename on the server."""
        import io

        data = aiohttp.FormData()
        data.add_field("image", io.BytesIO(image_data), filename=filename, content_type="image/png")
        if subfolder:
            data.add_field("subfolder", subfolder)
        if overwrite:
            data.add_field("overwrite", "true")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/upload/image",
                data=data,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Image upload failed (HTTP {resp.status}): {error_text}")
                result = await resp.json()
                uploaded_name = result.get("name", filename)
                logger.info(f"Uploaded image bytes as: {uploaded_name}")
                return uploaded_name

