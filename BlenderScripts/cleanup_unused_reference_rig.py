import argparse
import json
import sys
from pathlib import Path

import bpy


DEFAULT_REMOVE = [
    "MS_Mouse.001",
    "DeformationSystem.001",
]


def args_after_dash():
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--remove-object", action="append", default=[])
    parser.add_argument("--purge-orphans", action="store_true")
    return parser.parse_args(args_after_dash())


def main():
    args = parse_args()
    names = args.remove_object or DEFAULT_REMOVE
    removed = []
    skipped = []
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            skipped.append({"name": name, "reason": "missing"})
            continue
        if not obj.hide_viewport and not obj.hide_render:
            skipped.append({"name": name, "reason": "not_hidden"})
            continue
        data_name = obj.data.name if getattr(obj, "data", None) else None
        obj_type = obj.type
        bpy.data.objects.remove(obj, do_unlink=True)
        removed.append({"name": name, "type": obj_type, "data": data_name})

    if args.purge_orphans:
        for _ in range(3):
            bpy.ops.outliner.orphans_purge(do_recursive=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
    print("MOUSE_GEO_FORGE_CLEANUP_UNUSED_BEGIN")
    print(
        json.dumps(
            {
                "output": str(output),
                "removed": removed,
                "skipped": skipped,
            },
            indent=2,
        )
    )
    print("MOUSE_GEO_FORGE_CLEANUP_UNUSED_END")


if __name__ == "__main__":
    main()
