from dataclasses import dataclass, field
from typing import Optional
import time 

@dataclass
class Resume:
    id: Optional[int]
    file_path: str
    text_content: str
    vector_id: Optional[int]
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "file_path": self.file_path,
            "text_content": self.text_content,
            "vector_id": self.vector_id,
            "created_at": self.created_at
        }