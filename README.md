# Redoing bobo with CMake and Vulkan

## Getting Started

#### Step 1: Make sure you have the latest release versions of CMake and Vulkan installed.

[Windows CMake Download](https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-windows-x86_64.msi)

[Windows Vulkan SDK Download](https://vulkan.lunarg.com/sdk/home#windows)

When downloading Vulkan you must select the x64 debugger and memory allocator as addons.

![Vulkan additional options](image-3.png)

CMake must also be in your PATH environment variable.

![CMake Env Variable](image-2.png)

#### Step 2: Clone the repo and build.

```bash
git clone https://github.com/metalpipemomo/bobo-boogaloo.git
build.bat
```

The project solution should be auto-generated and will be opened up on first successful run of `build.bat`. You can run `build.bat` in Command Prompt or double click it in file select.

#### Step 3: Run the program.

You can run the program using the "Local Windows Debugger" button, same as bobo. Below is what a successful run should show.
![Successful run](image-4.png)

## If there are issues.

- Make sure your versions of CMake and Vulkan are correct.
- Ensure you have the correct addons for the Vulkan SDK.
- Ensure that CMake is in your PATH environment variable.

Run `clean.bat` before running a fresh `build.bat` to clean your build folder before re-building.
