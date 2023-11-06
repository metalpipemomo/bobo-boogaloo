# Redoing bobo with CMake and Vulkan

## Getting Started

#### Step 1: Make sure you have the latest release versions of CMake and Vulkan installed.

[Windows CMake Download](https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-windows-x86_64.msi)

[Windows Vulkan SDK Download](https://vulkan.lunarg.com/sdk/home#windows)

When downloading Vulkan you must select the x64 debugger and memory allocator as addons.

![Vulkan additional options](https://cdn.discordapp.com/attachments/579145370529955895/1171206457690181642/image-3.png?ex=655bd603&is=65496103&hm=6a25fd79890c35856e0563c6386aff77f4a2ebaf77668cde1784ce669dac4fe3&)

CMake must also be in your PATH environment variable.

![CMake Env Variable](https://cdn.discordapp.com/attachments/579145370529955895/1171206457421733909/image-2.png?ex=655bd603&is=65496103&hm=df67447bba6ecbf2bc10c0e3d328dffe32221c3d3eb022c9d2d315bb8cd1fdba&)

#### Step 2: Clone the repo and build.

```bash
git clone https://github.com/metalpipemomo/bobo-boogaloo.git
build.bat
```

The project solution should be auto-generated and will be opened up on first successful run of `build.bat`. You can run `build.bat` in Command Prompt or double click it in file select.

#### Step 3: Run the program.

You can run the program using the "Local Windows Debugger" button, same as bobo. Below is what a successful run should show.
![Successful run](https://media.discordapp.net/attachments/579145370529955895/1171206458000556104/image-4.png?ex=655bd603&is=65496103&hm=3c50dafbf286dba6de82ac283a2b85b18408532f68d76a90a2877bdaa1bc5ff4&=&width=1502&height=548)

## If there are issues.

- Make sure your versions of CMake and Vulkan are correct.
- Ensure you have the correct addons for the Vulkan SDK.
- Ensure that CMake is in your PATH environment variable.

Run `clean.bat` before running a fresh `build.bat` to clean your build folder before re-building.
