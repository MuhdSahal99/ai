from dataclasses import dataclass, field
from typing import Optional
import time 

@dataclass
class JobDescription:
    text_content: str
    original_filename: str
    id: int = field(default_factory=lambda: int(time.time() * 1000))
    vector_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "text_content": self.text_content,
            "vector_id": self.vector_id,
            "created_at": self.created_at
        }

def process_job_description(text_content: str, original_filename: str) -> JobDescription:
    return JobDescription(text_content=text_content, original_filename=original_filename)