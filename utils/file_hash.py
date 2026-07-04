import io
import hashlib
from typing import Union

def compute_file_hash(source: Union[str, io.BytesIO]) -> str:
    """Compute the SHA256 hash of the entire uploaded file contents."""
    hasher = hashlib.sha256()
    if isinstance(source, str):
        with open(source, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    elif hasattr(source, "read"):
        tell_pos = None
        if hasattr(source, "tell"):
            tell_pos = source.tell()
            source.seek(0)
        
        while chunk := source.read(8192):
            hasher.update(chunk)
            
        if tell_pos is not None and hasattr(source, "seek"):
            source.seek(tell_pos)
    else:
        hasher.update(source)
    return hasher.hexdigest()
