# arm64-osx, release builds only, with a fixed deployment target: msentity
# uses floating-point std::to_chars, which needs macOS 13.3 or later. Release
# wheels are tagged macosx_14_0_arm64 to match.
set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)
set(VCPKG_OSX_DEPLOYMENT_TARGET 14.0)
set(VCPKG_BUILD_TYPE release)
