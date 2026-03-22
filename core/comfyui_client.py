"""
ComfyUI Client — handles all communication with the ComfyUI server.

Async HTTP client for submitting workflows, monitoring progress,
uploading images, and retrieving generated results.
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Callable

import aiohttp

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Async client for ComfyUI API + WebSocket communication."""

    def __init__(self, config: dict):
        self.base_url = config.get("comfyui", {}).get("url", "http://127.0.0.1:8188")
        self.server_addr = self.base_url.replace("http://", "").replace("https://", "")
        self.client_id = f"ai_director_{uuid.uuid4().hex[:8]}"

    async def is_alive(self) -> bool:
        """Check if ComfyUI server is running."""
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
        """Submit a workflow to ComfyUI and return the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"ComfyUI rejected workflow: {error_text}")
                data = await resp.json()
                prompt_id = data["prompt_id"]
                logger.info(f"Workflow queued: {prompt_id}")
                return prompt_id

    async def wait_for_result(
        self,
        prompt_id: str,
        on_progress: Optional[Callable] = None,
        timeout: int = 300,
    ) -> dict:
        """
        Wait for a queued workflow to complete via WebSocket.
        
        Args:
            prompt_id: The prompt ID from queue_prompt
            on_progress: Optional callback(node_id, progress_value)
            timeout: Max seconds to wait
        
        Returns:
            Dict of node_id → list of image bytes
        """
        import websockets
        
        ws_url = f"ws://{self.server_addr}/ws?clientId={self.client_id}"
        
        try:
            async with websockets.connect(ws_url) as ws:
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > timeout:
                        raise TimeoutError(f"Generation timed out after {timeout}s")
                    
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    
                    if isinstance(msg, str):
                        data = json.loads(msg)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "progress" and on_progress:
                            node = data.get("data", {}).get("node", "")
                            value = data.get("data", {}).get("value", 0)
                            max_val = data.get("data", {}).get("max", 0)
                            await on_progress(node, value, max_val)
                        
                        elif msg_type == "executing":
                            exec_data = data.get("data", {})
                            if (
                                exec_data.get("node") is None
                                and exec_data.get("prompt_id") == prompt_id
                            ):
                                # Execution complete
                                break
                        
                        elif msg_type == "execution_error":
                            error_data = data.get("data", {})
                            raise RuntimeError(
                                f"ComfyUI execution error: {error_data.get('exception_message', 'Unknown error')}"
                            )
        except ImportError:
            # Fallback to polling if websockets not available
            logger.warning("websockets package not available, falling back to polling")
            return await self._poll_for_result(prompt_id, timeout)
        
        # Retrieve output images from history
        return await self._get_output_images(prompt_id)

    async def _poll_for_result(self, prompt_id: str, timeout: int = 300) -> dict:
        """Fallback polling method when websockets aren't available."""
        start = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start < timeout:
            history = await self.get_history(prompt_id)
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("completed", False) or history[prompt_id].get("outputs"):
                    return await self._get_output_images(prompt_id)
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI execution failed: {status}")
            await asyncio.sleep(2)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")

    async def _get_output_images(self, prompt_id: str) -> dict:
        """Retrieve generated images from ComfyUI history."""
        history = await self.get_history(prompt_id)
        
        if prompt_id not in history:
            raise RuntimeError(f"Prompt {prompt_id} not found in history")
        
        outputs = history[prompt_id].get("outputs", {})
        result = {}
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                images = []
                for img_info in node_output["images"]:
                    image_data = await self.get_image(
                        img_info["filename"],
                        img_info.get("subfolder", ""),
                        img_info.get("type", "output"),
                    )
                    images.append(image_data)
                result[node_id] = images
        
        return result

    async def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/history/{prompt_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return await resp.json()

    async def get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Download a generated image from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/view",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.read()

    async def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI and return the remote filename."""
        path = Path(image_path)
        logger.info(f"Uploading {path.name} to ComfyUI...")
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                open(image_path, "rb"),
                filename=path.name,
                content_type="image/png",
            )
            
            async with session.post(
                f"{self.base_url}/upload/image",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                result = await resp.json()
                remote_name = result["name"]
                logger.info(f"Uploaded as: {remote_name}")
                return remote_name

    async def upload_image_bytes(self, image_bytes: bytes, filename: str = "input.png") -> str:
        """Upload image bytes directly to ComfyUI."""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                image_bytes,
                filename=filename,
                content_type="image/png",
            )
            
            async with session.post(
                f"{self.base_url}/upload/image",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                result = await resp.json()
                return result["name"]

    async def get_queue_status(self) -> dict:
        """Get current ComfyUI queue status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/queue",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                data = await resp.json()
                return {
                    "running": len(data.get("queue_running", [])),
                    "pending": len(data.get("queue_pending", [])),
                }

    async def interrupt(self):
        """Interrupt current generation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/interrupt",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
