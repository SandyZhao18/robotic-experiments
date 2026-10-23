"""Read-only tour of every resource on the machine. Nothing here actuates."""

import asyncio

from viam.components.camera import Camera
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.services.vision import VisionClient

from robotic_experiments.connection import connect

# (resource class, name, read-only method to call).
# Generic components/services are left out on purpose: their only method is
# do_command, which may actuate.
CHECKS = [
    (Camera, "audience-cam", "get_images"),
    (Camera, "cam", "get_images"),
    (Arm, "arm", "get_end_position"),
    (Gripper, "gripper", "is_moving"),
    (Gripper, "table", "is_moving"),
    (Gripper, "wall-front", "is_moving"),
    (Gripper, "wall-side", "is_moving"),
    (Gripper, "ceiling", "is_moving"),
    (VisionClient, "hand-yolo", "get_properties"),
    (VisionClient, "hand-3d", "get_properties"),
    (VisionClient, "audience-gestures", "get_properties"),
]


async def inspect() -> None:
    async with await connect() as machine:
        print("Resources:")
        for rn in sorted(machine.resource_names, key=lambda rn: (rn.type, rn.subtype, rn.name)):
            print(f"  {rn.type}:{rn.subtype}/{rn.name}")

        available = {(rn.subtype, rn.name) for rn in machine.resource_names}
        for cls, name, method in CHECKS:
            if (cls.API.resource_subtype, name) not in available:
                print(f"{name}: not on machine, skipping")
                continue
            resource = cls.from_robot(machine, name)
            try:
                value = await getattr(resource, method)()
            except Exception as e:
                print(f"{name} {method} failed: {e!r}")
                continue
            print(f"{name} {method} return value: {value}")


def main() -> None:
    asyncio.run(inspect())


if __name__ == "__main__":
    main()
