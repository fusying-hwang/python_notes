import subprocess
from pathlib import Path
from typing import List

from h2hlib import MetabitProject, Package


def _get_proto_file_names() -> List[str]:
    result = []
    for proto_file in Path("interface").rglob("*.proto"):
        result.append(str(proto_file)[: -len(".proto")])
    return result
