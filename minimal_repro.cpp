// Minimal repro: vkInvalidateMappedMemoryRanges on one thread returns stale data
// when a concurrent vkFlushMappedMemoryRanges on another thread targets a
// disjoint range of the SAME VkDeviceMemory. Observed on MoltenVK / AMD Polaris
// / macOS 15. Per Vulkan spec, disjoint ranges of the same allocation may be
// flushed / invalidated concurrently from different threads.
//
// Build:   c++ -std=c++17 -O2 minimal_repro.cpp -lvulkan -o minimal_repro
//          (also needs glslc to compile the inline shader below; or pre-supply
//          a SPIR-V blob — see SHADER_SPV_PATH).
//
// Run:     VK_LAYER_PATH=$(brew --prefix vulkan-validationlayers)/share/vulkan/explicit_layer.d \
//          ./minimal_repro
//
// Result on the affected stack: thousands of mismatches per 10000 iterations.
// On a working stack: 0/10000.

#include <vulkan/vulkan.h>

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#define VK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK fail %s = %d\n", #expr, _r); std::exit(2); } } while (0)

// Compute shader: out[i] = in[i] ^ 0xA5A5A5A5. Compiled offline to SPIR-V.
#ifndef SHADER_SPV_PATH
#  define SHADER_SPV_PATH "build/shader.comp.spv"
#endif

static constexpr uint32_t kElems    = 1024;
static constexpr uint32_t kFrames   = 2;       // pipelined submits
static constexpr uint32_t kIters    = 10000;
static constexpr VkDeviceSize kBufSz = kElems * sizeof(uint32_t);

struct Frame {
    VkBuffer bufA, bufB;
    VkDeviceSize offsetA, offsetB;        // offsets in shared VkDeviceMemory
    void *mappedA, *mappedB;
    VkCommandBuffer cmd;
    VkFence fence;
    VkDescriptorSet ds;
    bool busy = false;   // protected by `mu` below
    uint32_t base = 0;
};

static std::vector<char> slurp(const char* p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "open %s\n", p); std::exit(2); }
    auto n = f.tellg(); f.seekg(0); std::vector<char> v(n); f.read(v.data(), n); return v;
}

static VkDeviceSize alignUp(VkDeviceSize x, VkDeviceSize a) {
    return (x + a - 1) & ~(a - 1);
}

int main() {
    // ---- Instance ----
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;

    // Validation layer is optional — set REPRO_VALIDATION=1 in the env to enable.
    // The bug reproduces with or without it; default off so the repro works on
    // installations that don't have the LunarG layers handy.
    std::vector<const char*> layers;
    if (std::getenv("REPRO_VALIDATION")) {
        const char* want = "VK_LAYER_KHRONOS_validation";
        uint32_t n = 0; vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> avail(n);
        vkEnumerateInstanceLayerProperties(&n, avail.data());
        for (auto& l : avail) if (std::strcmp(l.layerName, want) == 0) { layers.push_back(want); break; }
        if (layers.empty()) std::fprintf(stderr, "validation requested but %s not present; continuing without\n", want);
        else                std::fprintf(stderr, "validation: %s enabled\n", want);
    }
    const char* instExts[] = {
#ifdef __APPLE__
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        "VK_KHR_get_physical_device_properties2",
#endif
    };
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = sizeof(instExts)/sizeof(instExts[0]);
    ici.ppEnabledExtensionNames = instExts;
#ifdef __APPLE__
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    VkInstance instance; VK(vkCreateInstance(&ici, nullptr, &instance));

    // ---- Physical + queue family ----
    uint32_t pdN = 1; VkPhysicalDevice pd; VK(vkEnumeratePhysicalDevices(instance, &pdN, &pd));
    VkPhysicalDeviceProperties pdp; vkGetPhysicalDeviceProperties(pd, &pdp);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    uint32_t qfN; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfN);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, qfs.data());
    uint32_t qfi = ~0u;
    for (uint32_t i = 0; i < qfN; ++i) if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }

    // ---- Device ----
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char* devExts[] = {
#ifdef __APPLE__
        "VK_KHR_portability_subset",
#endif
    };
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = sizeof(devExts)/sizeof(devExts[0]);
    dci.ppEnabledExtensionNames = devExts;
    VkDevice device; VK(vkCreateDevice(pd, &dci, nullptr, &device));
    VkQueue queue; vkGetDeviceQueue(device, qfi, 0, &queue);

    // ---- Memory type: DEVICE_LOCAL+HOST_VISIBLE+HOST_CACHED, NOT HOST_COHERENT ----
    uint32_t memType = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)  &&
           !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { memType = i; break; }
    }
    if (memType == ~0u) { std::fprintf(stderr, "no non-coherent device-local host-cached type\n"); return 3; }

    // ---- Buffers + ONE shared VkDeviceMemory ----
    auto mkbuf = [&](VkDeviceSize sz) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = sz; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer b; VK(vkCreateBuffer(device, &bci, nullptr, &b)); return b;
    };

    Frame frames[kFrames];
    const VkDeviceSize atom = pdp.limits.nonCoherentAtomSize;
    const VkDeviceSize bufBytes = alignUp(kBufSz, atom);

    for (auto& fr : frames) { fr.bufA = mkbuf(bufBytes); fr.bufB = mkbuf(bufBytes); }
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, frames[0].bufA, &req);
    const VkDeviceSize stride = alignUp(req.size, std::max<VkDeviceSize>(req.alignment, atom));

    VkDeviceSize cursor = 0;
    for (auto& fr : frames) { fr.offsetA = cursor; cursor += stride; fr.offsetB = cursor; cursor += stride; }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = alignUp(cursor, atom); mai.memoryTypeIndex = memType;
    VkDeviceMemory mem; VK(vkAllocateMemory(device, &mai, nullptr, &mem));
    void* base; VK(vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &base));
    for (auto& fr : frames) {
        VK(vkBindBufferMemory(device, fr.bufA, mem, fr.offsetA));
        VK(vkBindBufferMemory(device, fr.bufB, mem, fr.offsetB));
        fr.mappedA = static_cast<char*>(base) + fr.offsetA;
        fr.mappedB = static_cast<char*>(base) + fr.offsetB;
    }

    // ---- Pipeline ----
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = b;
    VkDescriptorSetLayout dsl; VK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl));
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl; VK(vkCreatePipelineLayout(device, &plci, nullptr, &pl));

    auto spv = slurp(SHADER_SPV_PATH);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule sm; VK(vkCreateShaderModule(device, &smci, nullptr, &sm));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipeline; VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline));

    // ---- Per-frame cmd buf, fence, descriptor set ----
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kFrames};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = kFrames; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool; VK(vkCreateDescriptorPool(device, &dpci, nullptr, &dpool));

    VkCommandPoolCreateInfo cmdpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdpci.queueFamilyIndex = qfi; cmdpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; VK(vkCreateCommandPool(device, &cmdpci, nullptr, &cpool));

    for (auto& fr : frames) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        VK(vkAllocateDescriptorSets(device, &dsai, &fr.ds));
        VkDescriptorBufferInfo dbiA{fr.bufA, 0, kBufSz}, dbiB{fr.bufB, 0, kBufSz};
        VkWriteDescriptorSet w[2]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = fr.ds; w[0].dstBinding = 0;
        w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &dbiA;
        w[1] = w[0]; w[1].dstBinding = 1; w[1].pBufferInfo = &dbiB;
        vkUpdateDescriptorSets(device, 2, w, 0, nullptr);

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VK(vkAllocateCommandBuffers(device, &cbai, &fr.cmd));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK(vkCreateFence(device, &fci, nullptr, &fr.fence));
    }

    // ---- Reader thread: pulls frame indices off a queue, invalidates B, compares ----
    std::queue<uint32_t> jobs; std::mutex mu; std::condition_variable cv;
    bool stop = false;
    uint32_t bad = 0;

    std::thread reader([&]() {
        while (true) {
            uint32_t idx;
            { std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&]{ return stop || !jobs.empty(); });
              if (jobs.empty()) return;
              idx = jobs.front(); jobs.pop(); }
            Frame& fr = frames[idx];
            VK(vkWaitForFences(device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = mem; r.offset = fr.offsetB; r.size = bufBytes;
            VK(vkInvalidateMappedMemoryRanges(device, 1, &r));    // <-- the suspect call

            const uint32_t* B = static_cast<const uint32_t*>(fr.mappedB);
            uint32_t local_bad = 0;
            for (uint32_t i = 0; i < kElems; ++i) {
                uint32_t expect = (fr.base + i) ^ 0xA5A5A5A5u;
                if (B[i] != expect) { local_bad = 1; break; }
            }
            { std::lock_guard<std::mutex> g(mu); fr.busy = false; bad += local_bad; }
            cv.notify_all();
        }
    });

    // ---- Main loop ----
    for (uint32_t it = 0; it < kIters; ++it) {
        uint32_t idx = it % kFrames; Frame& fr = frames[idx];

        { std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&]{ return !fr.busy; }); }

        fr.base = it * 0x01010101u;
        uint32_t* A = static_cast<uint32_t*>(fr.mappedA);
        for (uint32_t i = 0; i < kElems; ++i) A[i] = fr.base + i;

        // Record command buffer.
        VK(vkResetCommandBuffer(fr.cmd, 0));
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(fr.cmd, &cbbi));
        VkBufferMemoryBarrier pre[2]{};
        pre[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        pre[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        pre[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        pre[0].srcQueueFamilyIndex = pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[0].buffer = fr.bufA; pre[0].size = VK_WHOLE_SIZE;
        pre[1] = pre[0]; pre[1].srcAccessMask = 0; pre[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pre[1].buffer = fr.bufB;
        vkCmdPipelineBarrier(fr.cmd, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 2, pre, 0, nullptr);
        vkCmdBindPipeline(fr.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(fr.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &fr.ds, 0, nullptr);
        vkCmdDispatch(fr.cmd, (kElems + 63) / 64, 1, 1);
        VkBufferMemoryBarrier post{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; post.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        post.srcQueueFamilyIndex = post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post.buffer = fr.bufB; post.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(fr.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &post, 0, nullptr);
        VK(vkEndCommandBuffer(fr.cmd));

        // Flush A's range (host -> device).  This call races with the reader's
        // invalidate-of-B-on-the-same-VkDeviceMemory in the previous iteration.
        VkMappedMemoryRange fr_range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        fr_range.memory = mem; fr_range.offset = fr.offsetA; fr_range.size = bufBytes;
        VK(vkFlushMappedMemoryRanges(device, 1, &fr_range));

        VK(vkResetFences(device, 1, &fr.fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &fr.cmd;
        VK(vkQueueSubmit(queue, 1, &si, fr.fence));

        { std::lock_guard<std::mutex> g(mu); fr.busy = true; jobs.push(idx); } cv.notify_one();
    }

    // Drain.
    { std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&]{
          if (!jobs.empty()) return false;
          for (auto& fr : frames) if (fr.busy) return false;
          return true; });
      stop = true; }
    cv.notify_all();
    reader.join();

    std::printf("device: %s  nonCoherentAtomSize=%llu\n",
                pdp.deviceName, (unsigned long long)atom);
    std::printf("RESULT: %u / %u iterations bad\n", bad, kIters);
    return bad == 0 ? 0 : 1;
}
