
# Script issues
#### Viser import "No module named 'websockets.asyncio'
If you are running any script that inherits `xi/base.py` and a line importing viser after launching an isaacsim app results in

```bash
ModuleNotFoundError: No module named 'websockets.asyncio'
```
delete the websockets package from the environment's `site-packages/isaacsim/extscache/omni.kit.pip_archive-0.0.0+10a4b5c0.lx64.cp310/pip_prebundle` directory. The exact path for the websockets package can be found by setting a breakpoint after isaacsim app launch:

```
# Isaacsim app launch
import argparse
from omni.isaac.lab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="This script tests YuMi with random jitter in a physics environment.")
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
args_cli.headless = True # Comment out if not headless devel and you'd like the interactive isaacsim gui
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pdb; pdb.set_trace()
import websockets
websockets.__file__
```

This is due to the fact that isaaclab's app launch alters the system path to search for packages in extscache first and points the websockets imports to isaacsim's directory instead. Isaacsim's websockets version is 12.0 whereas viser requires >=13.1 for its asyncio internal library.

#### [Error] [carb] Failed to create change watch for `some/file/path`: errno=28/No space left on device
https://forums.developer.nvidia.com/t/since-2022-version-error-failed-to-create-change-watch-no-space-left-on-device/218198