// Vulkan flush/invalidate corruption repro.
// See vulkan-flush-invalidate-repro-plan.md for hypothesis and matrix.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
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
    BarrierType barrier = BarrierType::Buffer;
    MemoryLayout layout = MemoryLayout::Shared;
    uint32_t iterations = 10000;
    bool verbose = false;
};

static const uint32_t kElems = 1024;
static const VkDeviceSize kBufferSize = kElems * sizeof(uint32_t);

static const char* barrierName(BarrierType b) {
    switch (b) {
        case BarrierType::Buffer: return "BUFFER";
        case BarrierType::Memory: return "MEMORY";
        case BarrierType::None:   return "NONE";
    }
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
        } else if (a == "--verbose") {
            c.verbose = true;
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: repro [--barrier=BUFFER|MEMORY|NONE] [--layout=SHARED|SEPARATE]\n"
                "             [--iterations=N] [--verbose]\n");
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            std::exit(1);
        }
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

// ---- Validation layer callback -------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    const char* sev =
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" :
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ? "WARNING" :
        (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) ? "INFO" : "VERBOSE";
    std::fprintf(stderr, "[validation %s] %s\n", sev, data->pMessage);
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)) {
        // Don't abort — just be loud. The repro itself decides pass/fail.
    }
    return VK_FALSE;
}

// ---- Helpers -------------------------------------------------------------------
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

int main(int argc, char** argv) {
    Config cfg = parseArgs(argc, argv);

    std::printf("== Vulkan flush/invalidate repro ==\n");
    std::printf("variant: barrier=%s layout=%s iterations=%u\n",
                barrierName(cfg.barrier), layoutName(cfg.layout), cfg.iterations);

    // ---- Instance --------------------------------------------------------------
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vulkan-flush-invalidate-repro";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "none";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation" };
    std::vector<const char*> instExts = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
#ifdef USE_PORTABILITY_ENUMERATION
    instExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instExts.push_back("VK_KHR_get_physical_device_properties2");
#endif

    // Verify validation layer is present; downgrade gracefully if not.
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

    // Debug messenger.
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    auto vkCreateDebugUtilsMessenger =
        (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    auto vkDestroyDebugUtilsMessenger =
        (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (vkCreateDebugUtilsMessenger) {
        VkDebugUtilsMessengerCreateInfoEXT mci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mci.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mci.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mci.pfnUserCallback = debugCb;
        VK_CHECK(vkCreateDebugUtilsMessenger(instance, &mci, nullptr, &messenger));
    }

    // ---- Physical device -------------------------------------------------------
    uint32_t pdCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, nullptr));
    if (pdCount == 0) { std::fprintf(stderr, "no physical devices\n"); return 2; }
    std::vector<VkPhysicalDevice> pds(pdCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, pds.data()));
    VkPhysicalDevice pd = pds[0];

    VkPhysicalDeviceProperties pdp{};
    vkGetPhysicalDeviceProperties(pd, &pdp);
    std::printf("device: %s\n", pdp.deviceName);
    std::printf("nonCoherentAtomSize: %llu\n",
                (unsigned long long)pdp.limits.nonCoherentAtomSize);

    // Memory type table dump.
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    std::printf("memory types:\n");
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        std::printf("  [%u] heap=%u flags=%s\n", i, mp.memoryTypes[i].heapIndex,
                    memFlagsToString(mp.memoryTypes[i].propertyFlags));
    }

    // ---- Queue family ----------------------------------------------------------
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    }
    if (qfi == UINT32_MAX) { std::fprintf(stderr, "no compute queue\n"); return 2; }

    // ---- Device ----------------------------------------------------------------
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

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
    VkQueue queue;
    vkGetDeviceQueue(device, qfi, 0, &queue);

    // ---- Pick memory type ------------------------------------------------------
    // We want DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED, NOT HOST_COHERENT.
    const VkMemoryPropertyFlags wantSet =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags wantUnset = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    uint32_t chosenMemType = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & wantSet) == wantSet && (f & wantUnset) == 0) {
            chosenMemType = i; break;
        }
    }
    if (chosenMemType == UINT32_MAX) {
        std::fprintf(stderr,
            "FATAL: no memory type with DEVICE_LOCAL+HOST_VISIBLE+HOST_CACHED (and not HOST_COHERENT)\n"
            "This repro is specifically about that path.\n");
        return 3;
    }
    std::printf("chosen memory type: [%u] %s\n",
                chosenMemType,
                memFlagsToString(mp.memoryTypes[chosenMemType].propertyFlags));

    const VkDeviceSize atom = pdp.limits.nonCoherentAtomSize;
    const VkDeviceSize bufBytes = alignUp(kBufferSize, atom);

    // ---- Buffers + memory ------------------------------------------------------
    auto makeBuffer = [&](VkDeviceSize size) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf;
        VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &buf));
        return buf;
    };

    VkBuffer bufA = makeBuffer(bufBytes);
    VkBuffer bufB = makeBuffer(bufBytes);

    VkMemoryRequirements reqA, reqB;
    vkGetBufferMemoryRequirements(device, bufA, &reqA);
    vkGetBufferMemoryRequirements(device, bufB, &reqB);

    if (!(reqA.memoryTypeBits & (1u << chosenMemType)) ||
        !(reqB.memoryTypeBits & (1u << chosenMemType))) {
        std::fprintf(stderr, "FATAL: chosen memory type not allowed for these buffers\n");
        return 3;
    }

    VkDeviceMemory memShared = VK_NULL_HANDLE;
    VkDeviceMemory memA = VK_NULL_HANDLE, memB = VK_NULL_HANDLE;
    void* mappedSharedBase = nullptr;
    void* mappedA = nullptr;
    void* mappedB = nullptr;
    VkDeviceSize offsetA = 0, offsetB = 0;
    VkDeviceSize sharedSize = 0;

    if (cfg.layout == MemoryLayout::Shared) {
        // One allocation, both buffers in it.
        // offset of B must be aligned to max(reqB.alignment, atom) so flush/invalidate ranges are clean.
        VkDeviceSize alignB = std::max<VkDeviceSize>(reqB.alignment, atom);
        offsetA = 0;
        offsetB = alignUp(reqA.size, alignB);
        sharedSize = offsetB + reqB.size;
        sharedSize = alignUp(sharedSize, atom);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = sharedSize;
        mai.memoryTypeIndex = chosenMemType;
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &memShared));

        VK_CHECK(vkBindBufferMemory(device, bufA, memShared, offsetA));
        VK_CHECK(vkBindBufferMemory(device, bufB, memShared, offsetB));

        VK_CHECK(vkMapMemory(device, memShared, 0, VK_WHOLE_SIZE, 0, &mappedSharedBase));
        mappedA = static_cast<char*>(mappedSharedBase) + offsetA;
        mappedB = static_cast<char*>(mappedSharedBase) + offsetB;

        std::printf("layout: SHARED  size=%llu  offsetA=%llu offsetB=%llu  atom=%llu\n",
                    (unsigned long long)sharedSize,
                    (unsigned long long)offsetA,
                    (unsigned long long)offsetB,
                    (unsigned long long)atom);
    } else {
        VkMemoryAllocateInfo maiA{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        maiA.allocationSize = alignUp(reqA.size, atom);
        maiA.memoryTypeIndex = chosenMemType;
        VK_CHECK(vkAllocateMemory(device, &maiA, nullptr, &memA));
        VK_CHECK(vkBindBufferMemory(device, bufA, memA, 0));
        VK_CHECK(vkMapMemory(device, memA, 0, VK_WHOLE_SIZE, 0, &mappedA));

        VkMemoryAllocateInfo maiB{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        maiB.allocationSize = alignUp(reqB.size, atom);
        maiB.memoryTypeIndex = chosenMemType;
        VK_CHECK(vkAllocateMemory(device, &maiB, nullptr, &memB));
        VK_CHECK(vkBindBufferMemory(device, bufB, memB, 0));
        VK_CHECK(vkMapMemory(device, memB, 0, VK_WHOLE_SIZE, 0, &mappedB));

        std::printf("layout: SEPARATE  sizeA=%llu sizeB=%llu  atom=%llu\n",
                    (unsigned long long)maiA.allocationSize,
                    (unsigned long long)maiB.allocationSize,
                    (unsigned long long)atom);
    }

    // ---- Pipeline / descriptors -----------------------------------------------
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

    auto spv = readFile(SHADER_SPV_PATH);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size();
    smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &sm));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &ds));

    VkDescriptorBufferInfo dbiA{bufA, 0, kBufferSize};
    VkDescriptorBufferInfo dbiB{bufB, 0, kBufferSize};
    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ds;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &dbiA;
    writes[1] = writes[0];
    writes[1].dstBinding = 1;
    writes[1].pBufferInfo = &dbiB;
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

    // ---- Command buffer + fence ------------------------------------------------
    VkCommandPoolCreateInfo cpoolci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpoolci.queueFamilyIndex = qfi;
    cpoolci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool;
    VK_CHECK(vkCreateCommandPool(device, &cpoolci, nullptr, &cpool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

    // ---- Iteration loop --------------------------------------------------------
    uint32_t bad = 0;
    uint32_t mismatchExamplesPrinted = 0;
    const uint32_t kMaxExamples = 5;
    std::vector<uint32_t> badIters;
    badIters.reserve(64);

    for (uint32_t it = 0; it < cfg.iterations; ++it) {
        // Host writes pattern into A.
        const uint32_t base = it * 0x01010101u;
        uint32_t* a = static_cast<uint32_t*>(mappedA);
        for (uint32_t i = 0; i < kElems; ++i) {
            a[i] = base + i;
        }

        // Build command buffer fresh each iteration.
        VK_CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd, &cbbi));

        // Pre-dispatch barrier: host write -> shader read on A; also ensure B is in shader_write state.
        if (cfg.barrier == BarrierType::Buffer) {
            VkBufferMemoryBarrier bmb[2] = {};
            bmb[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bmb[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bmb[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bmb[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb[0].buffer = bufA;
            bmb[0].offset = 0;
            bmb[0].size = VK_WHOLE_SIZE;
            bmb[1] = bmb[0];
            bmb[1].srcAccessMask = 0;
            bmb[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bmb[1].buffer = bufB;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 2, bmb, 0, nullptr);
        } else if (cfg.barrier == BarrierType::Memory) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);
        }
        // BarrierType::None: rely on implicit submit-time host-write availability.

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdDispatch(cmd, (kElems + 63) / 64, 1, 1);

        // Post-dispatch barrier: shader write -> host read on B.
        if (cfg.barrier == BarrierType::Buffer) {
            VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bmb.buffer = bufB;
            bmb.offset = 0;
            bmb.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &bmb, 0, nullptr);
        } else if (cfg.barrier == BarrierType::Memory) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &mb, 0, nullptr, 0, nullptr);
        }

        VK_CHECK(vkEndCommandBuffer(cmd));

        // Flush A's range (host -> device).
        {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            if (cfg.layout == MemoryLayout::Shared) {
                r.memory = memShared;
                r.offset = offsetA;
                r.size   = bufBytes;
            } else {
                r.memory = memA;
                r.offset = 0;
                r.size = VK_WHOLE_SIZE;
            }
            VK_CHECK(vkFlushMappedMemoryRanges(device, 1, &r));
        }

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkResetFences(device, 1, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        // Invalidate B's range (device -> host).
        {
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            if (cfg.layout == MemoryLayout::Shared) {
                r.memory = memShared;
                r.offset = offsetB;
                r.size   = bufBytes;
            } else {
                r.memory = memB;
                r.offset = 0;
                r.size = VK_WHOLE_SIZE;
            }
            VK_CHECK(vkInvalidateMappedMemoryRanges(device, 1, &r));
        }

        // Compare.
        const uint32_t* b = static_cast<const uint32_t*>(mappedB);
        bool iterBad = false;
        uint32_t firstBadIdx = 0;
        uint32_t firstExpected = 0, firstActual = 0;
        for (uint32_t i = 0; i < kElems; ++i) {
            uint32_t expected = (base + i) ^ 0xA5A5A5A5u;
            if (b[i] != expected) {
                if (!iterBad) {
                    iterBad = true;
                    firstBadIdx = i;
                    firstExpected = expected;
                    firstActual = b[i];
                }
            }
        }
        if (iterBad) {
            ++bad;
            if (badIters.size() < 64) badIters.push_back(it);
            if (mismatchExamplesPrinted < kMaxExamples) {
                std::printf("  MISMATCH iter=%u idx=%u expected=0x%08x actual=0x%08x\n",
                            it, firstBadIdx, firstExpected, firstActual);
                ++mismatchExamplesPrinted;
            }
        }
        if (cfg.verbose && (it % 1000 == 0)) {
            std::printf("  ... iter %u / %u  bad=%u\n", it, cfg.iterations, bad);
        }
    }

    std::printf("RESULT: %u / %u iterations bad\n", bad, cfg.iterations);
    if (!badIters.empty()) {
        std::printf("first bad iterations:");
        for (size_t i = 0; i < badIters.size() && i < 32; ++i) std::printf(" %u", badIters[i]);
        std::printf("\n");
    }

    // ---- Cleanup ---------------------------------------------------------------
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cpool, nullptr);
    vkDestroyDescriptorPool(device, dpool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, sm, nullptr);
    vkDestroyPipelineLayout(device, pl, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);

    if (cfg.layout == MemoryLayout::Shared) {
        vkUnmapMemory(device, memShared);
        vkDestroyBuffer(device, bufA, nullptr);
        vkDestroyBuffer(device, bufB, nullptr);
        vkFreeMemory(device, memShared, nullptr);
    } else {
        vkUnmapMemory(device, memA);
        vkUnmapMemory(device, memB);
        vkDestroyBuffer(device, bufA, nullptr);
        vkDestroyBuffer(device, bufB, nullptr);
        vkFreeMemory(device, memA, nullptr);
        vkFreeMemory(device, memB, nullptr);
    }

    vkDestroyDevice(device, nullptr);
    if (messenger && vkDestroyDebugUtilsMessenger) {
        vkDestroyDebugUtilsMessenger(instance, messenger, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    return bad == 0 ? 0 : 1;
}
