MODULE_NAME_TABLE = {
    "p3767-0005": "NVIDIA Jetson Orin Nano (Developer kit)",
    "p3767-0004": "NVIDIA Jetson Orin Nano (4GB ram)",
    "p3767-0003": "NVIDIA Jetson Orin Nano (8GB ram)",
    "p3767-0001": "NVIDIA Jetson Orin NX (8GB ram)",
    "p3767-0000": "NVIDIA Jetson Orin NX (16GB ram)",
    "p3701-0005": "NVIDIA Jetson AGX Orin (64GB ram)",
    "p3701-0004": "NVIDIA Jetson AGX Orin (32GB ram)",
    "p3701-0002": "NVIDIA Jetson IGX Orin (Developer kit)",
    "p3701-0000": "NVIDIA Jetson AGX Orin",
    "p3668-0003": "NVIDIA Jetson Xavier NX (16GB ram)",
    "p3668-0001": "NVIDIA Jetson Xavier NX",
    "p3668-0000": "NVIDIA Jetson Xavier NX (Developer kit)",
    "p2888-0008": "NVIDIA Jetson AGX Xavier Industrial (32 GB ram)",
    "p2888-0006": "NVIDIA Jetson AGX Xavier (8 GB ram)",
    "p2888-0005": "NVIDIA Jetson AGX Xavier (64 GB ram)",
    "p2888-0004": "NVIDIA Jetson AGX Xavier (32 GB ram)",
    "p2888-0003": "NVIDIA Jetson AGX Xavier (32 GB ram)",
    "p2888-0001": "NVIDIA Jetson AGX Xavier (16 GB ram)",
    "p3448-0003": "NVIDIA Jetson Nano (2 GB ram)",
    "p3448-0002": "NVIDIA Jetson Nano module (16Gb eMMC)",
    "p3448-0000": "NVIDIA Jetson Nano (4 GB ram)",
    "p3636-0001": "NVIDIA Jetson TX2 NX",
    "p3509-0000": "NVIDIA Jetson TX2 NX",
    "p3489-0888": "NVIDIA Jetson TX2 (4 GB ram)",
    "p3489-0000": "NVIDIA Jetson TX2i",
    "p3310-1000": "NVIDIA Jetson TX2",
    "p2180-1000": "NVIDIA Jetson TX1",
    "r375-0001": "NVIDIA Jetson TK1",
    "p3904-0000": "NVIDIA Clara AGX",
    # Other modules
    "p2595-0000-A0": "Nintendo Switch",
}

from pathlib import Path


def get_module_name() -> str:
    try:
        with open("/proc/device-tree/nvidia,dtsfilename", "r") as file:
            board_name = Path(file.read().strip()).stem
        module_name = "Unknown board"
        for module, name in MODULE_NAME_TABLE.items():
            if module in board_name:
                module_name = name
                break
        return module_name
    except FileNotFoundError:
        return "Unknown board"
