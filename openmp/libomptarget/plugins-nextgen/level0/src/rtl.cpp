//===----RTLs/amdgpu/src/rtl.cpp - Target RTLs Implementation ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for L0 machine
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cassert>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "Utilities.h"
#include "omptarget.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "ze_api.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

struct L0KernelTy;
struct L0DeviceTy;
struct L0PluginTy;
struct L0StreamTy;
struct L0EventTy;
struct L0StreamManagerTy;
struct L0EventManagerTy;
struct L0MemoryManagerTy;
struct L0MemoryPoolTy;

template <typename ResourceTy>
struct L0ResourceRef : public GenericDeviceResourceRef {
  /// Create an empty reference to an invalid resource.
  L0ResourceRef() : Resource(nullptr) {}

  /// Create a reference to an existing resource.
  L0ResourceRef(ResourceTy *Resource) : Resource(Resource) {}

  virtual ~L0ResourceRef() {}

  /// Create a new resource and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override;

  /// Destroy the referenced resource and invalidate the reference. The
  /// reference must be to a valid event before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    // TODO
  }

  /// Get the underlying L0SignalTy reference.
  operator ResourceTy *() const { return Resource; }

private:
  /// The reference to the actual resource.
  ResourceTy *Resource;
};

/// Class implementing the L0 kernel functionalities which derives from the
/// generic kernel class.
struct L0KernelTy : public GenericKernelTy {
  /// Create an L0 kernel with a name and an execution mode.
  L0KernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode)
      : GenericKernelTy(Name, ExecutionMode) {}

  /// Initialize the L0 kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    assert(0);
  }

  /// Launch the L0 kernel function.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    assert(0);
  }

  /// Print more elaborate kernel launch info for L0
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                               KernelArgsTy &KernelArgs, uint32_t NumThreads,
                               uint64_t NumBlocks) const override {
    assert(0);
  }

  /// The default number of blocks is common to the whole device.
  uint32_t getDefaultNumBlocks(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultNumBlocks();
  }

  /// The default number of threads is common to the whole device.
  uint32_t getDefaultNumThreads(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultNumThreads();
  }

  /// Get group and private segment kernel size.
  uint32_t getGroupSize() const { return GroupSize; }
  uint32_t getPrivateSize() const { return PrivateSize; }

  /// Get the HSA kernel object representing the kernel function.
  uint64_t getKernelObject() const { return KernelObject; }

private:
  /// The kernel object to execute.
  uint64_t KernelObject;

  /// The args, group and private segments sizes required by a kernel instance.
  uint32_t ArgsSize;
  uint32_t GroupSize;
  uint32_t PrivateSize;

  /// The size of implicit kernel arguments.
  // const uint32_t ImplicitArgsSize;

  /// Additional Info for the AMD GPU Kernel
  // std::optional<utils::KernelMetaDataTy> KernelInfo;
};

struct L0SignalTy {
  /// Create an empty signal.
  L0SignalTy() : UseCount() {}
  L0SignalTy(L0DeviceTy &Device) : UseCount() {}

private:
  /// The underlying HSA signal.
  // hsa_signal_t Signal;

  /// Reference counter for tracking the concurrent use count. This is mainly
  /// used for knowing how many streams are using the signal.
  RefCountTy<> UseCount;
};

/// Classes for holding L0 signals and managing signals.
using L0SignalRef = L0ResourceRef<L0SignalTy>;
using L0SignalManagerTy = GenericDeviceResourceManagerTy<L0SignalRef>;

struct L0QueueTy {
  L0QueueTy() {}
};

/// Abstract class that holds the common members of the actual kernel devices
/// and the host device. Both types should inherit from this class.
struct L0GenericDeviceTy {
  L0GenericDeviceTy() {}

  virtual ~L0GenericDeviceTy() {}
};

/// Class representing the host device. This host device may have more than one
/// HSA host agent. We aggregate all its resources into the same instance.
struct L0HostDeviceTy : public L0GenericDeviceTy {
  /// Create a host device from an array of host agents.
  L0HostDeviceTy() : L0GenericDeviceTy() {}
};

/// Class implementing the L0 device functionalities
struct L0DeviceTy : public GenericDeviceTy {
  // Create an L0 device with a device id and default L0 grid values.
  L0DeviceTy(int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(DeviceId, NumDevices, SPIR64GridValues64),
        L0StreamManager(*this), L0EventManager(*this), L0SignalManager(*this) {}

  ~L0DeviceTy() {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    auto deviceId = getDeviceId();

    return Plugin::success();
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override { assert(0); }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const override {
    assert(0);
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override { return ComputeUnitKind; }

  /// Allocate and construct an L0 kernel.
  Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) override {
    assert(0);
  }

  Error setContext() override { assert(0); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    assert(0);
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    assert(0);
  }

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override { assert(0); }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override { assert(0); }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override { assert(0); }

  /// Pin the host buffer and return the device pointer that should be used for
  /// device transfers.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    assert(0);
  }

  /// Unpin the host buffer.
  Error dataUnlockImpl(void *HstPtr) override { assert(0); }

  /// Check through the HSA runtime whether the \p HstPtr buffer is pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    assert(0);
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Exchange data between two devices within the plugin.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Initialize the device info for interoperability purposes.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    assert(0);
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override { assert(0); }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *EventPtr) override { assert(0); }

  /// Record the event.
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    assert(0);
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override { assert(0); }

  /// Print information about the device.
  Error printInfoImpl() override { return Plugin::success(); }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }
  Error setDeviceStackSize(uint64_t Value) override {
    return Plugin::success();
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }
  Error setDeviceHeapSize(uint64_t Value) override { return Plugin::success(); }

  /// Get the signal manager.
  L0SignalManagerTy &getSignalManager() { return L0SignalManager; }

private:
  using L0StreamRef = L0ResourceRef<L0StreamTy>;
  using L0EventRef = L0ResourceRef<L0EventTy>;
  using L0StreamManagerTy = GenericDeviceResourceManagerTy<L0StreamRef>;
  using L0EventManagerTy = GenericDeviceResourceManagerTy<L0EventRef>;

  /// Handle towards the device
  ze_device_handle_t deviceHandle;

  /// Stream manager for L0 streams.
  L0StreamManagerTy L0StreamManager;

  /// Event manager for L0 events.
  L0EventManagerTy L0EventManager;

  /// Signal manager for L0 signals.
  L0SignalManagerTy L0SignalManager;

  /// The GPU architecture.
  std::string ComputeUnitKind;

  /// List of device packet queues.
  std::vector<L0QueueTy> Queues;
};

/// Class implementing the L0-specific functionalities of the global
/// handler.
struct L0GlobalHandlerTy final : public GenericGlobalHandlerTy {
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    assert(0);
  }

private:
  /// Extract the global's information from the ELF image, section, and symbol.
  Error getGlobalMetadataFromELF(const DeviceImageTy &Image,
                                 const ELF64LE::Sym &Symbol,
                                 const ELF64LE::Shdr &Section,
                                 GlobalTy &ImageGlobal) override {
    // The global's address in L0 is computed as the image begin + the ELF
    // symbol value. Notice we do not add the ELF section offset.
    ImageGlobal.setPtr(advanceVoidPtr(Image.getStart(), Symbol.st_value));

    // Set the global's size.
    ImageGlobal.setSize(Symbol.st_size);

    return Plugin::success();
  }
};

/// Class implementing the Level Zero-specific functionalities of the plugin.
struct L0PluginTy final : public GenericPluginTy {
  /// Create an L0 plugin and initialize the L0 driver.
  L0PluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  L0PluginTy(const L0PluginTy &) = delete;
  L0PluginTy(L0PluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    ze_result_t Status;
    uint32_t driverCount = 0, totalDeviceCount = 0;
    Status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (Status != ZE_RESULT_SUCCESS) {
      DP("Failed to initialize the Level Zero library\n");
      return 0;
    }

    Status = zeDriverGet(&driverCount, NULL);
    if (Status != ZE_RESULT_SUCCESS) {
      DP("Failed to query the Level Zero driver count\n");
      return 0;
    }

    driverHandles =
        (ze_driver_handle_t *)malloc(driverCount * sizeof(ze_driver_handle_t));
    Status = zeDriverGet(&driverCount, driverHandles);
    if (Status != ZE_RESULT_SUCCESS) {
      DP("Failed to query the Level Zero drivers\n");
      return 0;
    }

    for (uint32_t driver_idx = 0; driver_idx < driverCount; driver_idx++) {
      ze_driver_handle_t driver = driverHandles[driver_idx];

      // if count is zero, then the driver will update the value with the total
      // number of devices available.
      uint32_t deviceCount = 0;
      Status = zeDeviceGet(driver, &deviceCount, NULL);
      if (Status != ZE_RESULT_SUCCESS) {
        DP("Failed to query the Level Zero device count from driver\n");
        return 0;
      }

      ze_device_handle_t *deviceHandles = (ze_device_handle_t *)malloc(
          deviceCount * sizeof(ze_device_handle_t));

      Status = zeDeviceGet(driver, &deviceCount, deviceHandles);
      if (Status != ZE_RESULT_SUCCESS) {
        DP("Failed to query the Level Zero devices from driver\n");
        return 0;
      }

      for (uint32_t device_idx = 0; device_idx < deviceCount; device_idx++) {
        ze_device_handle_t device = deviceHandles[device_idx];

        ze_device_properties_t device_properties{};
        device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        Status = zeDeviceGetProperties(device, &device_properties);
        if (Status != ZE_RESULT_SUCCESS) {
          DP("Failed to query the Level Zero device properties\n");
          return 0;
        }

        if (device_properties.type == ZE_DEVICE_TYPE_GPU) {
          devices.push_back(device);
        }
      }

      free(deviceHandles);
    }

    return static_cast<int32_t>(devices.size());
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    if (driverHandles)
      free(driverHandles);

    return Plugin::success();
  }

  Triple::ArchType getTripleArch() const override { return Triple::spir64; }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { return 0; }

  /// Check whether the image is compatible with a L0 device.
  Expected<bool> isImageCompatible(__tgt_image_info *Info) const override {
    // TODO Figure out what can be in Info->Arch
    return true;
  }

  /// TODO does this plugin does support exchanging data between two devices ?
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return false;
  }

private:
  ze_driver_handle_t *driverHandles;
  llvm::SmallVector<ze_device_handle_t> devices;
};

GenericPluginTy *Plugin::createPlugin() { return new L0PluginTy(); }

GenericDeviceTy *Plugin::createDevice(int32_t DeviceId, int32_t NumDevices) {
  // TODO
}

GenericGlobalHandlerTy *Plugin::createGlobalHandler() {
  // TODO
}

template <typename... ArgsTy>
Error Plugin::check(int32_t Code, const char *ErrFmt, ArgsTy... Args) {
  // TODO
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
