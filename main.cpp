// Vulkan flush/invalidate corruption repro.
// See ../museld/vulkan-flush-invalidate-repro-plan.md for hypothesis and matrix.
//
// Variants exposed via flags:
//   --barrier=BUFFER|MEMORY|NONE         barrier kind around the dispatch
//   --layout=SHARED|SEPARATE             one VkDeviceMemory or per-buffer
//   --frames-in-flight=N                 N>1 means pipelined submits (no fence wait between iters)
//   --threaded-reader                    separate host thread does invalidate+compare of B
//   --neighbours=N                       (SHARED only) N filler buffers between A and B per frame
//   --cmd-split                          split work into two CBs in one submit
//   --two-queues                         CB1 on queue 0, CB2 on queue 1 (requires queueCount>=2)
//   --iterations=N                       total iterations
//   --verbose                            periodic progress
//
// The defaults reproduce the original 4-variant matrix when combined with --barrier and --layout.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define VK_CHECK(expr) do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
        std::fprintf(stderr, "VK call failed: %s = %d at %s:%d\n", #expr, _r, __FILE__, __LINE__); \
        std::exit(2); \
    } \
} while (0)

enum class BarrierType { Buffer, Memory, None };
enum class MemoryLayout { Shared, Separate };

struct Config {
    BarrierType  barrier        = BarrierType::Buffer;
    MemoryLayout layout         = MemoryLayout::Shared;
    uint32_t     iterations     = 10000;
    uint32_t     framesInFlight = 1;
    uint32_t     neighbours     = 0;
    bool         threadedReader = false;
    bool         cmdSplit       = false;
    bool         twoQueues      = false;
    bool         verbose        = false;
};

static const uint32_t kElems = 1024;
static const VkDeviceSize kBufferSize = kElems * sizeof(uint32_t);

static const char* barrierName(BarrierType b) {
    switch (b) { case BarrierType::Buffer: return "BUFFER";
                 case BarrierType::Memory: return "MEMORY";
                 case BarrierType::None:   return "NONE"; }
    return "?";
}
static const char* layoutName(MemoryLayout l) {
    return l == MemoryLayout::Shared ? "SHARED" : "SEPARATE";
}

static Config parseArgs(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto eat = [&](const std::string& key, std::string& out) -> bool {
            if (a.rfind(key, 0) == 0) { out = a.substr(key.size()); return true; }
            return false;
        };
        std::string v;
        if (eat("--barrier=", v)) {
            if      (v == "BUFFER") c.barrier = BarrierType::Buffer;
            else if (v == "MEMORY") c.barrier = BarrierType::Memory;
            else if (v == "NONE")   c.barrier = BarrierType::None;
            else { std::fprintf(stderr, "bad --barrier=%s\n", v.c_str()); std::exit(1); }
        } else if (eat("--layout=", v)) {
            if      (v == "SHARED")   c.layout = MemoryLayout::Shared;
            else if (v == "SEPARATE") c.layout = MemoryLayout::Separate;
            else { std::fprintf(stderr, "bad --layout=%s\n", v.c_str()); std::exit(1); }
        } else if (eat("--iterations=", v)) {
            c.iterations = static_cast<uint32_t>(std::stoul(v));
        } else if (eat("--frames-in-flight=", v)) {
            c.framesInFlight = static_cast<uint32_t>(std::stoul(v));
            if (c.framesInFlight == 0) c.framesInFlight = 1;
        } else if (eat("--neighbours=", v)) {
            c.neighbours = static_cast<uint32_t>(std::stoul(v));
        } else if (a == "--threaded-reader") c.threadedReader = true;
        else if  (a == "--cmd-split")       c.cmdSplit = true;
        else if  (a == "--two-queues")      c.twoQueues = true;
        else if  (a == "--verbose")         c.verbose = true;
        else if  (a == "--help" || a == "-h") {
            std::printf(
                "Usage: repro [--barrier=BUFFER|MEMORY|NONE] [--layout=SHARED|SEPARATE]\n"
                "             [--frames-in-flight=N] [--neighbours=N]\n"
                "             [--threaded-reader] [--cmd-split] [--two-queues]\n"
                "             [--iterations=N] [--verbose]\n");
            std::exit(0);
        } else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); std::exit(1); }
    }
    return c;
}

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(n);
    f.read(buf.data(), n);
    return buf;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    const char* sev =
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" :
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARNING" :
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) ? "INFO" : "VERBOSE";
    std::fprintf(stderr, "[validation %s] %s\n", sev, data->pMessage);
    return VK_FALSE;
}

static VkDeviceSize alignUp(VkDeviceSize x, VkDeviceSize a) {
    return (x + a - 1) & ~(a - 1);
}

static const char* memFlagsToString(VkMemoryPropertyFlags f) {
    static thread_local std::string s;
    s.clear();
    auto add = [&](const char* n) { if (!s.empty()) s += "|"; s += n; };
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)     add("DEVICE_LOCAL");
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)     add("HOST_VISIBLE");
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)    add("HOST_COHERENT");
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)      add("HOST_CACHED");
    if (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) add("LAZILY_ALLOCATED");
    if (s.empty()) s = "0";
    return s.c_str();
}

// ---- Per-frame resources ------------------------------------------------------
struct Frame {
    VkBuffer bufA = VK_NULL_HANDLE, bufB = VK_NULL_HANDLE;
    std::vector<VkBuffer> nb;                   // neighbour filler buffers (shared layout only)
    VkDeviceSize offsetA = 0, offsetB = 0;      // offsets in shared memory (or 0 for separate)
    VkDeviceMemory memA = VK_NULL_HANDLE;       // separate layout only
    VkDeviceMemory memB = VK_NULL_HANDLE;       // separate layout only
    void* mappedA = nullptr;
    void* mappedB = nullptr;
    VkCommandBuffer cmd  = VK_NULL_HANDLE;
    VkCommandBuffer cmd2 = VK_NULL_HANDLE;      // cmd-split only
    VkSemaphore     sem  = VK_NULL_HANDLE;      // cmd-split / two-queues handshake
    VkFence         fence = VK_NULL_HANDLE;     // signaled by the LAST submit of this frame
    VkDescriptorSet ds   = VK_NULL_HANDLE;

    // Pipeline state (multi-thread coordination):
    //   READY      -> main thread may use it for a new submission
    //   SUBMITTED  -> reader thread should wait fence + read
    enum class State { Ready, Submitted };
    std::atomic<State> state{State::Ready};
    uint32_t expectedBase = 0;
};

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    if (cfg.neighbours > 0 && cfg.layout != MemoryLayout::Shared) {
        std::fprintf(stderr, "--neighbours requires --layout=SHARED\n"); return 1;
    }

    std::printf("== Vulkan flush/invalidate repro ==\n");
    std::printf("variant: barrier=%s layout=%s frames=%u neighbours=%u%s%s%s iterations=%u\n",
                barrierName(cfg.barrier), layoutName(cfg.layout),
                cfg.framesInFlight, cfg.neighbours,
                cfg.threadedReader ? " threaded-reader" : "",
                cfg.cmdSplit       ? " cmd-split" : "",
                cfg.twoQueues      ? " two-queues" : "",
                cfg.iterations);

    // ---- Instance --------------------------------------------------------------
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vulkan-flush-invalidate-repro";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    std::vector<const char*> instExts = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
#ifdef USE_PORTABILITY_ENUMERATION
    instExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instExts.push_back("VK_KHR_get_physical_device_properties2");
#endif

    {
        uint32_t n = 0;
        vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> avail(n);
        vkEnumerateInstanceLayerProperties(&n, avail.data());
        bool found = false;
        for (auto& l : avail) if (std::strcmp(l.layerName, layers[0]) == 0) found = true;
        if (!found) {
            std::fprintf(stderr, "WARNING: %s not available; running without validation\n", layers[0]);
            layers.clear();
        }
    }

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &appInfo;
    ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = static_cast<uint32_t>(instExts.size());
    ici.ppEnabledExtensionNames = instExts.data();
#ifdef USE_PORTABILITY_ENUMERATION
    ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    auto vkCreateDebugUtilsMessenger =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    auto vkDestroyDebugUtilsMessenger =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (vkCreateDebugUtilsMessenger) {
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mci.pfnUserCallback = debugCb;
        VK_CHECK(vkCreateDebugUtilsMessenger(instance, &mci, nullptr, &messenger));
    }

    // ---- Physical device + queue family ---------------------------------------
    uint32_t pdCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, nullptr));
    if (pdCount == 0) { std::fprintf(stderr, "no physical devices\n"); return 2; }
    std::vector<VkPhysicalDevice> pds(pdCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, pds.data()));
    VkPhysicalDevice pd = pds[0];

    VkPhysicalDeviceProperties pdp{};
    vkGetPhysicalDeviceProperties(pd, &pdp);
    std::printf("device: %s\n", pdp.deviceName);
    std::printf("nonCoherentAtomSize: %llu\n", (unsigned long long)pdp.limits.nonCoherentAtomSize);

    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    std::printf("memory types:\n");
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        std::printf("  [%u] heap=%u flags=%s\n", i, mp.memoryTypes[i].heapIndex,
                    memFlagsToString(mp.memoryTypes[i].propertyFlags));
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());

    uint32_t qfi = UINT32_MAX;
    uint32_t queuesWanted = cfg.twoQueues ? 2 : 1;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if ((qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qfs[i].queueCount >= queuesWanted) {
            qfi = i; break;
        }
    }
    if (qfi == UINT32_MAX) {
        if (cfg.twoQueues) {
            std::fprintf(stderr,
                "FATAL: --two-queues needs a compute queue family with queueCount>=2; none found\n");
            return 2;
        }
        std::fprintf(stderr, "no compute queue\n"); return 2;
    }
    std::printf("queue family %u with queueCount=%u\n", qfi, qfs[qfi].queueCount);

    // ---- Device + queues -------------------------------------------------------
    std::vector<float> prios(queuesWanted, 1.0f);
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = queuesWanted;
    qci.pQueuePriorities = prios.data();

    std::vector<const char*> devExts;
#ifdef USE_PORTABILITY_ENUMERATION
    devExts.push_back("VK_KHR_portability_subset");
#endif

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();

    VkDevice device;
    VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &device));

    VkQueue queue0, queue1 = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, qfi, 0, &queue0);
    if (cfg.twoQueues) vkGetDeviceQueue(device, qfi, 1, &queue1);

    // ---- Memory type -----------------------------------------------------------
    const VkMemoryPropertyFlags wantSet =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags wantUnset = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    uint32_t chosenMemType = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & wantSet) == wantSet && (f & wantUnset) == 0) { chosenMemType = i; break; }
    }
    if (chosenMemType == UINT32_MAX) {
        std::fprintf(stderr,
            "FATAL: no memory type with DEVICE_LOCAL+HOST_VISIBLE+HOST_CACHED (and not HOST_COHERENT)\n");
        return 3;
    }
    std::printf("chosen memory type: [%u] %s\n", chosenMemType,
                memFlagsToString(mp.memoryTypes[chosenMemType].propertyFlags));

    const VkDeviceSize atom    = pdp.limits.nonCoherentAtomSize;
    const VkDeviceSize bufBytes = alignUp(kBufferSize, atom);

    // ---- Per-frame resources ---------------------------------------------------
    auto makeBuffer = [&](VkDeviceSize size) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf;
        VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf));
        return buf;
    };

    std::vector<Frame> frames(cfg.framesInFlight);

    // Create the buffers up front so we can collect alignment/size requirements.
    for (auto& fr : frames) {
        fr.bufA = makeBuffer(bufBytes);
        fr.bufB = makeBuffer(bufBytes);
        fr.nb.resize(cfg.neighbours);
        for (auto& nb : fr.nb) nb = makeBuffer(bufBytes);
    }

    VkMemoryRequirements reqA{};
    vkGetBufferMemoryRequirements(device, frames[0].bufA, &reqA);
    if (!(reqA.memoryTypeBits & (1u << chosenMemType))) {
        std::fprintf(stderr, "FATAL: chosen memory type not allowed for this buffer\n");
        return 3;
    }

    VkDeviceMemory memShared = VK_NULL_HANDLE;
    void* mappedSharedBase = nullptr;
    VkDeviceSize sharedSize = 0;

    if (cfg.layout == MemoryLayout::Shared) {
        // Layout per frame: A | nb[0] | nb[1] | ... | B. Cursor walks one big block.
        const VkDeviceSize alignAny = std::max<VkDeviceSize>(reqA.alignment, atom);
        VkDeviceSize cursor = 0;
        for (auto& fr : frames) {
            fr.offsetA = cursor;
            cursor = alignUp(cursor + reqA.size, alignAny);
            for (auto& nb : fr.nb) {
                (void)nb;
                cursor = alignUp(cursor + reqA.size, alignAny);
            }
            fr.offsetB = cursor;
            cursor = alignUp(cursor + reqA.size, alignAny);
        }
        sharedSize = alignUp(cursor, atom);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = sharedSize;
        mai.memoryTypeIndex = chosenMemType;
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memShared));
        VK_CHECK(vkMapMemory(device, memShared, 0, VK_WHOLE_SIZE, 0, &mappedSharedBase));

        for (size_t f = 0; f < frames.size(); ++f) {
            auto& fr = frames[f];
            VK_CHECK(vkBindBufferMemory(device, fr.bufA, memShared, fr.offsetA));
            VK_CHECK(vkBindBufferMemory(device, fr.bufB, memShared, fr.offsetB));
            // Bind neighbours at fixed offsets between A and B.
            VkDeviceSize nbCursor = alignUp(fr.offsetA + reqA.size,
                                            std::max<VkDeviceSize>(reqA.alignment, atom));
            for (auto& nb : fr.nb) {
                VK_CHECK(vkBindBufferMemory(device, nb, memShared, nbCursor));
                nbCursor = alignUp(nbCursor + reqA.size,
                                   std::max<VkDeviceSize>(reqA.alignment, atom));
            }
            fr.mappedA = static_cast<char*>(mappedSharedBase) + fr.offsetA;
            fr.mappedB = static_cast<char*>(mappedSharedBase) + fr.offsetB;
        }
        std::printf("layout: SHARED total=%llu  per-frame stride approx %llu  atom=%llu\n",
                    (unsigned long long)sharedSize,
                    (unsigned long long)(frames.size() > 1 ? frames[1].offsetA - frames[0].offsetA : 0),
                    (unsigned long long)atom);
    } else {
        for (auto& fr : frames) {
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = alignUp(reqA.size, atom);
            mai.memoryTypeIndex = chosenMemType;
            VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &fr.memA));
            VK_CHECK(vkBindBufferMemory(device, fr.bufA, fr.memA, 0));
            VK_CHECK(vkMapMemory(device, fr.memA, 0, VK_WHOLE_SIZE, 0, &fr.mappedA));
            VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &fr.memB));
            VK_CHECK(vkBindBufferMemory(device, fr.bufB, fr.memB, 0));
            VK_CHECK(vkMapMemory(device, fr.memB, 0, VK_WHOLE_SIZE, 0, &fr.mappedB));
        }
        std::printf("layout: SEPARATE per-frame sizeA=%llu sizeB=%llu  atom=%llu\n",
                    (unsigned long long)alignUp(reqA.size, atom),
                    (unsigned long long)alignUp(reqA.size, atom),
                    (unsigned long long)atom);
    }

    // ---- Pipeline / descriptors -----------------------------------------------
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

    auto spv = readFile(SHADER_SPV_PATH);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &sm));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName  = "main";
    cpci.layout = pl;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * cfg.framesInFlight};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = cfg.framesInFlight;
    dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dpool));

    // Descriptor set + cmd buffers + fence + semaphore per frame.
    VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpoolci.queueFamilyIndex = qfi;
    cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool;
    VK_CHECK(vkCreateCommandPool(device, &cpoolci, nullptr, &cpool));

    for (auto& fr : frames) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &fr.ds));

        VkDescriptorBufferInfo dbiA{fr.bufA, 0, kBufferSize};
        VkDescriptorBufferInfo dbiB{fr.bufB, 0, kBufferSize};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = fr.ds; writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &dbiA;
        writes[1] = writes[0];
        writes[1].dstBinding = 1; writes[1].pBufferInfo = &dbiB;
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &fr.cmd));
        if (cfg.cmdSplit || cfg.twoQueues) {
            VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &fr.cmd2));
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &fr.sem));
        }
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &fr.fence));
    }

    // ---- Helpers to record one frame's command buffers ------------------------
    // Records into 'cmd' (and optionally 'cmd2' for split). When split, cmd holds
    // pre-barrier+dispatch, cmd2 holds the post-barrier.
    auto recordFrame = [&](Frame& fr) {
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        const bool split = cfg.cmdSplit || cfg.twoQueues;
        VkCommandBuffer firstCb = fr.cmd;
        VkCommandBuffer secondCb = split ? fr.cmd2 : fr.cmd;

        VK_CHECK(vkResetCommandBuffer(firstCb, 0));
        VK_CHECK(vkBeginCommandBuffer(firstCb, &cbbi));

        // Pre-dispatch barrier.
        if (cfg.barrier == BarrierType::Buffer) {
            VkBufferMemoryBarrier bmb[2] = {};
            bmb[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bmb[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bmb[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bmb[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[0].buffer = fr.bufA; bmb[0].offset = 0; bmb[0].size = VK_WHOLE_SIZE;
            bmb[1] = bmb[0];
            bmb[1].srcAccessMask = 0;
            bmb[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bmb[1].buffer = fr.bufB;
            vkCmdPipelineBarrier(firstCb,
                VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 2, bmb, 0, nullptr);
        } else if (cfg.barrier == BarrierType::Memory) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(firstCb,
                VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        vkCmdBindPipeline(firstCb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(firstCb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &fr.ds, 0, nullptr);
        vkCmdDispatch(firstCb, (kElems + 63) / 64, 1, 1);

        if (split) {
            VK_CHECK(vkEndCommandBuffer(firstCb));
            VK_CHECK(vkResetCommandBuffer(secondCb, 0));
            VK_CHECK(vkBeginCommandBuffer(secondCb, &cbbi));
        }

        if (cfg.barrier == BarrierType::Buffer) {
            VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb.buffer = fr.bufB; bmb.offset = 0; bmb.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(secondCb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &bmb, 0, nullptr);
        } else if (cfg.barrier == BarrierType::Memory) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(secondCb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        VK_CHECK(vkEndCommandBuffer(secondCb));
    };

    // Submit a frame's CB(s); fence is signaled by the LAST submit.
    auto submitFrame = [&](Frame& fr) {
        const bool split = cfg.cmdSplit || cfg.twoQueues;
        if (!split) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &fr.cmd;
            VK_CHECK(vkResetFences(device, 1, &fr.fence));
            VK_CHECK(vkQueueSubmit(queue0, 1, &si, fr.fence));
            return;
        }

        // Split: CB1 signals fr.sem; CB2 waits on it. Same queue (cmd-split) or different (two-queues).
        VkSubmitInfo s1{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        s1.commandBufferCount = 1; s1.pCommandBuffers = &fr.cmd;
        s1.signalSemaphoreCount = 1; s1.pSignalSemaphores = &fr.sem;

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        VkSubmitInfo s2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        s2.commandBufferCount = 1; s2.pCommandBuffers = &fr.cmd2;
        s2.waitSemaphoreCount = 1; s2.pWaitSemaphores = &fr.sem;
        s2.pWaitDstStageMask = &waitStage;

        VK_CHECK(vkResetFences(device, 1, &fr.fence));
        if (cfg.twoQueues) {
            VK_CHECK(vkQueueSubmit(queue0, 1, &s1, VK_NULL_HANDLE));
            VK_CHECK(vkQueueSubmit(queue1, 1, &s2, fr.fence));
        } else {
            VkSubmitInfo arr[2] = {s1, s2};
            VK_CHECK(vkQueueSubmit(queue0, 2, arr, fr.fence));
        }
    };

    auto flushA = [&](Frame& fr) {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        if (cfg.layout == MemoryLayout::Shared) {
            r.memory = memShared; r.offset = fr.offsetA; r.size = bufBytes;
        } else {
            r.memory = fr.memA; r.offset = 0; r.size = VK_WHOLE_SIZE;
        }
        VK_CHECK(vkFlushMappedMemoryRanges(device, 1, &r));
    };
    auto invalidateB = [&](Frame& fr) {
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        if (cfg.layout == MemoryLayout::Shared) {
            r.memory = memShared; r.offset = fr.offsetB; r.size = bufBytes;
        } else {
            r.memory = fr.memB; r.offset = 0; r.size = VK_WHOLE_SIZE;
        }
        VK_CHECK(vkInvalidateMappedMemoryRanges(device, 1, &r));
    };

    // ---- Reader thread coordination -------------------------------------------
    std::atomic<uint32_t> bad{0};
    std::atomic<uint32_t> mismatchExamplesPrinted{0};
    const uint32_t kMaxExamples = 5;
    std::mutex printMu;

    auto checkFrame = [&](Frame& fr, uint32_t iterIndex) {
        const uint32_t base = fr.expectedBase;
        const uint32_t* b = static_cast<const uint32_t*>(fr.mappedB);
        bool iterBad = false;
        uint32_t firstBadIdx = 0, firstExpected = 0, firstActual = 0;
        for (uint32_t i = 0; i < kElems; ++i) {
            uint32_t expected = (base + i) ^ 0xA5A5A5A5u;
            if (b[i] != expected) {
                if (!iterBad) { iterBad = true; firstBadIdx = i; firstExpected = expected; firstActual = b[i]; }
            }
        }
        if (iterBad) {
            uint32_t prev = bad.fetch_add(1);
            if (prev < kMaxExamples) {
                std::lock_guard<std::mutex> g(printMu);
                std::printf("  MISMATCH iter=%u idx=%u expected=0x%08x actual=0x%08x\n",
                            iterIndex, firstBadIdx, firstExpected, firstActual);
                mismatchExamplesPrinted.fetch_add(1);
            }
        }
    };

    // Reader thread: pulls (frameIdx, iterIndex) off a queue.
    struct Job { uint32_t frameIdx; uint32_t iterIndex; };
    std::queue<Job> jobs;
    std::mutex jobsMu;
    std::condition_variable jobsCv;
    std::atomic<bool> stop{false};

    std::thread reader;
    if (cfg.threadedReader) {
        reader = std::thread([&]() {
            while (true) {
                Job job;
                {
                    std::unique_lock<std::mutex> lk(jobsMu);
                    jobsCv.wait(lk, [&]{ return stop.load() || !jobs.empty(); });
                    if (jobs.empty() && stop.load()) return;
                    job = jobs.front(); jobs.pop();
                }
                Frame& fr = frames[job.frameIdx];
                VK_CHECK(vkWaitForFences(device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
                invalidateB(fr);
                checkFrame(fr, job.iterIndex);
                fr.state.store(Frame::State::Ready, std::memory_order_release);
                jobsCv.notify_all();   // wake up main thread waiting for this frame to free up
            }
        });
    }

    // ---- Main iteration loop ---------------------------------------------------
    for (uint32_t it = 0; it < cfg.iterations; ++it) {
        uint32_t frameIdx = it % cfg.framesInFlight;
        Frame& fr = frames[frameIdx];

        // Wait until this frame slot is available again.
        if (cfg.threadedReader) {
            std::unique_lock<std::mutex> lk(jobsMu);
            jobsCv.wait(lk, [&]{ return fr.state.load(std::memory_order_acquire) == Frame::State::Ready; });
        } else if (it >= cfg.framesInFlight) {
            // First framesInFlight iterations have nothing to wait on; afterwards, the slot's
            // previous submission must finish AND we must invalidate+check it before reuse.
            VK_CHECK(vkWaitForFences(device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
            invalidateB(fr);
            checkFrame(fr, it - cfg.framesInFlight);
        }

        const uint32_t base = it * 0x01010101u;
        fr.expectedBase = base;
        uint32_t* a = static_cast<uint32_t*>(fr.mappedA);
        for (uint32_t i = 0; i < kElems; ++i) a[i] = base + i;

        recordFrame(fr);
        flushA(fr);
        submitFrame(fr);

        if (cfg.threadedReader) {
            fr.state.store(Frame::State::Submitted, std::memory_order_release);
            {
                std::lock_guard<std::mutex> g(jobsMu);
                jobs.push({frameIdx, it});
            }
            jobsCv.notify_one();
        }

        if (cfg.verbose && (it % 1000 == 0)) {
            std::printf("  ... iter %u / %u  bad=%u\n", it, cfg.iterations, bad.load());
        }
    }

    // Drain remaining in-flight frames.
    if (cfg.threadedReader) {
        // Wait until the reader has processed every queued job AND every frame is Ready.
        {
            std::unique_lock<std::mutex> lk(jobsMu);
            jobsCv.wait(lk, [&]{
                if (!jobs.empty()) return false;
                for (auto& fr : frames)
                    if (fr.state.load() != Frame::State::Ready) return false;
                return true;
            });
            stop.store(true);
            jobsCv.notify_all();
        }
        reader.join();
    } else {
        // Pick up the last framesInFlight submissions that haven't been checked yet.
        for (uint32_t k = 0; k < cfg.framesInFlight && k < cfg.iterations; ++k) {
            uint32_t it = cfg.iterations - cfg.framesInFlight + k;
            uint32_t frameIdx = it % cfg.framesInFlight;
            Frame& fr = frames[frameIdx];
            VK_CHECK(vkWaitForFences(device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
            invalidateB(fr);
            checkFrame(fr, it);
        }
    }

    std::printf("RESULT: %u / %u iterations bad\n", bad.load(), cfg.iterations);

    // ---- Cleanup ---------------------------------------------------------------
    vkDeviceWaitIdle(device);

    for (auto& fr : frames) {
        vkDestroyFence(device, fr.fence, nullptr);
        if (fr.sem) vkDestroySemaphore(device, fr.sem, nullptr);
        for (auto nb : fr.nb) vkDestroyBuffer(device, nb, nullptr);
        vkDestroyBuffer(device, fr.bufA, nullptr);
        vkDestroyBuffer(device, fr.bufB, nullptr);
        if (fr.memA) { vkUnmapMemory(device, fr.memA); vkFreeMemory(device, fr.memA, nullptr); }
        if (fr.memB) { vkUnmapMemory(device, fr.memB); vkFreeMemory(device, fr.memB, nullptr); }
    }
    if (memShared) { vkUnmapMemory(device, memShared); vkFreeMemory(device, memShared, nullptr); }

    vkDestroyCommandPool(device, cpool, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, sm, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);

    vkDestroyDevice(device, nullptr);
    if (messenger && vkDestroyDebugUtilsMessenger)
        vkDestroyDebugUtilsMessenger(instance, messenger, nullptr);
    vkDestroyInstance(instance, nullptr);
    return bad.load() == 0 ? 0 : 1;
}
