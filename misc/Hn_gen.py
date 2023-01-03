from pathlib import Path

import tomlkit

configs_path = Path(__file__).parent.parent / "test" / "configs" / "scaling"

config_template = {
    "qmc": {
        "iterations": 3,
        "batch_size": 256,
    },
    "system": {
        "basis": "6-31g",
    },
}

for n in range(4, 32 + 1, 4):
    config = config_template.copy()
    config["system"]["spins"] = [n // 2, n // 2]
    config["system"]["atoms"] = [
        {"name": "H", "coord": [i * 2.0, 0.0, 0.0]} for i in range(n)
    ]
    with open(configs_path / f"H{n}.toml", "w") as f:
        tomlkit.dump(config, f)
