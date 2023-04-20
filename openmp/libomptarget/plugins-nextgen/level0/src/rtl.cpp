//===----RTLs/amdgpu/src/rtl.cpp - Target RTLs Implementation ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for AMDGPU machine
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

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Utility class representing generic resource references to AMDGPU resources.
template <typename ResourceTy>
struct AMDGPUResourceRef : public GenericDeviceResourceRef {
  /// Create an empty reference to an invalid resource.
  AMDGPUResourceRef() : Resource(nullptr) {}

  /// Create a reference to an existing resource.
  AMDGPUResourceRef(ResourceTy *Resource) : Resource(Resource) {}

  virtual ~AMDGPUResourceRef() {}

  /// Create a new resource and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override;

  /// Destroy the referenced resource and invalidate the reference. The
  /// reference must be to a valid event before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    // TODO
  }

  /// Get the underlying AMDGPUSignalTy reference.
  operator ResourceTy *() const { return Resource; }

private:
  /// The reference to the actual resource.
  ResourceTy *Resource;
};

/// Class implementing the AMDGPU device images' properties.
struct AMDGPUDeviceImageTy : public DeviceImageTy {
  /// Create the AMDGPU image with the id and the target image pointer.
  AMDGPUDeviceImageTy(int32_t ImageId, const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, TgtImage) {}

  /// Prepare and load the executable corresponding to the image.
  Error loadExecutable(const AMDGPUDeviceTy &Device);

  /// Unload the executable.
  Error unloadExecutable() {
    // TODO
  }

  /// Get the executable.
  hsa_executable_t getExecutable() const { return Executable; }

  /// Find an HSA device symbol by its name on the executable.
  Expected<hsa_executable_symbol_t>
  findDeviceSymbol(GenericDeviceTy &Device, StringRef SymbolName) const;

  /// Get additional info for kernel, e.g., register spill counts
  std::optional<utils::KernelMetaDataTy>
  getKernelInfo(StringRef Identifier) const {
    auto It = KernelInfoMap.find(Identifier);

    if (It == KernelInfoMap.end())
      return {};

    return It->second;
  }

private:
  StringMap<utils::KernelMetaDataTy> KernelInfoMap;
};

/// Class implementing the AMDGPU kernel functionalities which derives from the
/// generic kernel class.
struct AMDGPUKernelTy : public GenericKernelTy {
  /// Create an AMDGPU kernel with a name and an execution mode.
  AMDGPUKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode)
      : GenericKernelTy(Name, ExecutionMode),
        ImplicitArgsSize(sizeof(utils::AMDGPUImplicitArgsTy)) {}

  /// Initialize the AMDGPU kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    // TODO
  }

  /// Launch the AMDGPU kernel function.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  /// Print more elaborate kernel launch info for AMDGPU
  Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                               KernelArgsTy &KernelArgs, uint32_t NumThreads,
                               uint64_t NumBlocks) const override;

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
  const uint32_t ImplicitArgsSize;

  /// Additional Info for the AMD GPU Kernel
  std::optional<utils::KernelMetaDataTy> KernelInfo;
};

/// Abstract class that holds the common members of the actual kernel devices
/// and the host device. Both types should inherit from this class.
struct AMDGenericDeviceTy {
  AMDGenericDeviceTy() {}

  virtual ~AMDGenericDeviceTy() {}
};

/// Class representing the host device. This host device may have more than one
/// HSA host agent. We aggregate all its resources into the same instance.
struct AMDHostDeviceTy : public AMDGenericDeviceTy {
  /// Create a host device from an array of host agents.
  AMDHostDeviceTy(const llvm::SmallVector<hsa_agent_t> &HostAgents)
      : AMDGenericDeviceTy(), Agents(HostAgents), ArgsMemoryManager(),
        PinnedMemoryManager() {}

  /// Initialize the host device memory pools and the memory managers for
  /// kernel args and host pinned memory allocations.
  Error init() {
    // TODO
  }

  /// Deinitialize memory pools and managers.
  Error deinit() {
    // TODO
  }
};

/// Class implementing the AMDGPU device functionalities which derives from the
/// generic device class.
struct AMDGPUDeviceTy : public GenericDeviceTy, AMDGenericDeviceTy {
  // Create an AMDGPU device with a device id and default AMDGPU grid values.
  AMDGPUDeviceTy(int32_t DeviceId, int32_t NumDevices,
                 AMDHostDeviceTy &HostDevice, hsa_agent_t Agent)
      : GenericDeviceTy(DeviceId, NumDevices, {0}), AMDGenericDeviceTy(),
        OMPX_NumQueues("LIBOMPTARGET_AMDGPU_NUM_HSA_QUEUES", 4),
        OMPX_QueueSize("LIBOMPTARGET_AMDGPU_HSA_QUEUE_SIZE", 512),
        OMPX_DefaultTeamsPerCU("LIBOMPTARGET_AMDGPU_TEAMS_PER_CU", 4),
        OMPX_MaxAsyncCopyBytes("LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES",
                               1 * 1024 * 1024), // 1MB
        OMPX_InitialNumSignals("LIBOMPTARGET_AMDGPU_NUM_INITIAL_HSA_SIGNALS",
                               64),
        AMDGPUStreamManager(*this), AMDGPUEventManager(*this),
        AMDGPUSignalManager(*this), Agent(Agent), HostDevice(HostDevice),
        Queues() {}

  ~AMDGPUDeviceTy() {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    // TODO
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override {
    // TODO
  }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const override {
    // TODO
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override { return ComputeUnitKind; }

  /// Allocate and construct an AMDGPU kernel.
  Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) override {
    // TODO
  }

  Error setContext() override { // TODO
  }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // TODO
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override;

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    // TODO
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    // TODO
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    // TODO
  }

  /// Pin the host buffer and return the device pointer that should be used for
  /// device transfers.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    // TODO
  }

  /// Unpin the host buffer.
  Error dataUnlockImpl(void *HstPtr) override {
    // TODO
  }

  /// Check through the HSA runtime whether the \p HstPtr buffer is pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    // TODO
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Exchange data between two devices within the plugin.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Initialize the device info for interoperability purposes.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    // TODO
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override {
    // TODO
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *EventPtr) override {
    // TODO
  }

  /// Record the event.
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // TODO
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override {
    // TODO
  }

  /// Print information about the device.
  Error printInfoImpl() override {
    // TODO: Implement the basic info.
    return Plugin::success();
  }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    // TODO
  }
  Error setDeviceStackSize(uint64_t Value) override {
    // TODO
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    // TODO
  }
  Error setDeviceHeapSize(uint64_t Value) override {
    // TODO
  }

  /// Get the device agent.
  hsa_agent_t getAgent() const override { return Agent; }

  /// Get the signal manager.
  AMDGPUSignalManagerTy &getSignalManager() { return AMDGPUSignalManager; }

private:
  using AMDGPUStreamRef = AMDGPUResourceRef<AMDGPUStreamTy>;
  using AMDGPUEventRef = AMDGPUResourceRef<AMDGPUEventTy>;

  using AMDGPUStreamManagerTy = GenericDeviceResourceManagerTy<AMDGPUStreamRef>;
  using AMDGPUEventManagerTy = GenericDeviceResourceManagerTy<AMDGPUEventRef>;

  /// Envar for controlling the number of HSA queues per device. High number of
  /// queues may degrade performance.
  UInt32Envar OMPX_NumQueues;

  /// Envar for controlling the size of each HSA queue. The size is the number
  /// of HSA packets a queue is expected to hold. It is also the number of HSA
  /// packets that can be pushed into each queue without waiting the driver to
  /// process them.
  UInt32Envar OMPX_QueueSize;

  /// Envar for controlling the default number of teams relative to the number
  /// of compute units (CUs) the device has:
  ///   #default_teams = OMPX_DefaultTeamsPerCU * #CUs.
  UInt32Envar OMPX_DefaultTeamsPerCU;

  /// Envar specifying the maximum size in bytes where the memory copies are
  /// asynchronous operations. Up to this transfer size, the memory copies are
  /// asychronous operations pushed to the corresponding stream. For larger
  /// transfers, they are synchronous transfers.
  UInt32Envar OMPX_MaxAsyncCopyBytes;

  /// Envar controlling the initial number of HSA signals per device. There is
  /// one manager of signals per device managing several pre-allocated signals.
  /// These signals are mainly used by AMDGPU streams. If needed, more signals
  /// will be created.
  UInt32Envar OMPX_InitialNumSignals;

  /// Stream manager for AMDGPU streams.
  AMDGPUStreamManagerTy AMDGPUStreamManager;

  /// Event manager for AMDGPU events.
  AMDGPUEventManagerTy AMDGPUEventManager;

  /// Signal manager for AMDGPU signals.
  AMDGPUSignalManagerTy AMDGPUSignalManager;

  /// The agent handler corresponding to the device.
  hsa_agent_t Agent;

  /// The GPU architecture.
  std::string ComputeUnitKind;

  /// Reference to the host device.
  AMDHostDeviceTy &HostDevice;

  /// List of device packet queues.
  std::vector<AMDGPUQueueTy> Queues;
};

/// Class implementing the AMDGPU-specific functionalities of the global
/// handler.
struct AMDGPUGlobalHandlerTy final : public GenericGlobalHandlerTy {
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    // TODO
  }

private:
  /// Extract the global's information from the ELF image, section, and symbol.
  Error getGlobalMetadataFromELF(const DeviceImageTy &Image,
                                 const ELF64LE::Sym &Symbol,
                                 const ELF64LE::Shdr &Section,
                                 GlobalTy &ImageGlobal) override {
    // The global's address in AMDGPU is computed as the image begin + the ELF
    // symbol value. Notice we do not add the ELF section offset.
    ImageGlobal.setPtr(advanceVoidPtr(Image.getStart(), Symbol.st_value));

    // Set the global's size.
    ImageGlobal.setSize(Symbol.st_size);

    return Plugin::success();
  }
};

/// Class implementing the AMDGPU-specific functionalities of the plugin.
struct AMDGPUPluginTy final : public GenericPluginTy {
  /// Create an AMDGPU plugin and initialize the AMDGPU driver.
  AMDGPUPluginTy()
      : GenericPluginTy(getTripleArch()), Initialized(false),
        HostDevice(nullptr) {}

  /// This class should not be copied.
  AMDGPUPluginTy(const AMDGPUPluginTy &) = delete;
  AMDGPUPluginTy(AMDGPUPluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    // TODO
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    // TODO
  }

  Triple::ArchType getTripleArch() const override {
    // TODO
  }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override {
    // TODO
  }

  /// Check whether the image is compatible with an AMDGPU device.
  Expected<bool> isImageCompatible(__tgt_image_info *Info) const override {
    // TODO
  }

  /// This plugin does not support exchanging data between two devices.
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return false;
  }

  /// Get the host device instance.
  AMDHostDeviceTy &getHostDevice() {
    assert(HostDevice && "Host device not initialized");
    return *HostDevice;
  }

  /// Get the kernel agent with the corresponding agent id.
  hsa_agent_t getKernelAgent(int32_t AgentId) const {
    assert((uint32_t)AgentId < KernelAgents.size() && "Invalid agent id");
    return KernelAgents[AgentId];
  }

  /// Get the list of the available kernel agents.
  const llvm::SmallVector<hsa_agent_t> &getKernelAgents() const {
    return KernelAgents;
  }
};

Error AMDGPUKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                                 uint32_t NumThreads, uint64_t NumBlocks,
                                 KernelArgsTy &KernelArgs, void *Args,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  // TODO
}

Error AMDGPUKernelTy::printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                             KernelArgsTy &KernelArgs,
                                             uint32_t NumThreads,
                                             uint64_t NumBlocks) const {
  // TODO
}

GenericPluginTy *Plugin::createPlugin() { // TODO
}

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

void *AMDGPUMemoryManagerTy::allocate(size_t Size, void *HstPtr,
                                      TargetAllocTy Kind) {}

void *AMDGPUDeviceTy::allocate(size_t Size, void *, TargetAllocTy Kind) {}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
