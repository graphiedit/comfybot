"""
Queue Manager — handles multi-user job queuing with a single GPU.

Provides FIFO queue with priority boosts for interactions (upscale/vary),
status tracking per job, and callback notifications.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    BUILDING = "building"
    GENERATING = "generating"
    UPSCALING = "upscaling"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """A single generation job."""
    id: str = ""
    user_id: str = ""
    guild_id: str = ""
    channel_id: str = ""
    prompt: str = ""
    status: JobStatus = JobStatus.QUEUED
    priority: int = 0                       # Higher = processed first
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    result_images: list = field(default_factory=list)   # List of image bytes
    error: str = ""
    metadata: dict = field(default_factory=dict)        # Extra data (plan, seed, etc.)
    
    # Callback to notify when status changes
    on_status_change: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()


class QueueManager:
    """
    Async job queue for managing image generation requests.
    
    Processes one job at a time (single GPU), with status callbacks
    for real-time Discord updates.
    """

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active_jobs: dict[str, Job] = {}
        self._history: dict[str, Job] = {}
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._processor: Optional[Callable] = None
        self._current_count = 0

    def set_processor(self, processor: Callable):
        """
        Set the job processor function.
        
        The processor should be an async function that takes a Job
        and returns a list of image bytes.
        """
        self._processor = processor

    async def start(self):
        """Start the queue worker."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Queue manager started")

    async def stop(self):
        """Stop the queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Queue manager stopped")

    async def submit(self, job: Job) -> str:
        """
        Submit a job to the queue.
        
        Returns the job ID.
        """
        self._active_jobs[job.id] = job
        await self._queue.put(job)
        
        position = self._queue.qsize()
        logger.info(f"Job {job.id} queued (position: {position})")
        
        return job.id

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID (active or history)."""
        if job_id in self._active_jobs:
            return self._active_jobs[job_id]
        return self._history.get(job_id)

    def get_queue_position(self, job_id: str) -> int:
        """Get the position of a job in the queue (0 = being processed)."""
        # This is approximate since we can't peek into asyncio.Queue easily
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            if job.status in (JobStatus.GENERATING, JobStatus.ANALYZING, JobStatus.BUILDING):
                return 0
        return self._queue.qsize()

    def get_queue_size(self) -> int:
        """Get the number of jobs in the queue."""
        return self._queue.qsize()

    def get_active_count(self) -> int:
        """Get the number of actively processing jobs."""
        return self._current_count

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Cannot cancel jobs already generating."""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                await self._notify_status(job)
                self._finish_job(job)
                return True
        return False

    async def _worker(self):
        """Main worker loop — processes jobs from the queue."""
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            
            if job.status == JobStatus.CANCELLED:
                continue
            
            self._current_count += 1
            try:
                await self._process_job(job)
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                job.status = JobStatus.FAILED
                job.error = str(e)
                await self._notify_status(job)
            finally:
                self._current_count -= 1
                self._finish_job(job)

    async def _process_job(self, job: Job):
        """Process a single job through the generation pipeline."""
        if not self._processor:
            raise RuntimeError("No job processor set — call set_processor() first")
        
        job.started_at = time.time()
        job.status = JobStatus.ANALYZING
        await self._notify_status(job)
        
        # The processor handles the full pipeline:
        # analyze → build workflow → generate → return images
        result_images = await self._processor(job)
        
        job.result_images = result_images
        job.status = JobStatus.COMPLETE
        job.completed_at = time.time()
        
        elapsed = job.completed_at - job.started_at
        logger.info(f"Job {job.id} completed in {elapsed:.1f}s")
        
        await self._notify_status(job)

    async def _notify_status(self, job: Job):
        """Notify the status change callback."""
        if job.on_status_change:
            try:
                await job.on_status_change(job)
            except Exception as e:
                logger.error(f"Status callback failed for job {job.id}: {e}")

    def _finish_job(self, job: Job):
        """Move a job from active to history."""
        if job.id in self._active_jobs:
            del self._active_jobs[job.id]
        self._history[job.id] = job
        
        # Keep only last 100 jobs in history
        if len(self._history) > 100:
            oldest_key = next(iter(self._history))
            del self._history[oldest_key]
