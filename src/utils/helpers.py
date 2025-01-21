from typing import Any

def process_data(batch_size: int, num_workers: int, use_gpu: bool) -> Any:
    """
    Example data processing function.
    
    Args:
        batch_size: Size of processing batches
        num_workers: Number of workers to use
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Any: Processing result
    """
    return batch_size * num_workers
