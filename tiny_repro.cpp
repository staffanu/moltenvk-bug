// Tiniest repro of the MoltenVK / AMD / macOS cache-coherency bug.
// No compute, no shader, no descriptors. Just two buffers in one
// VkDeviceMemory, a GPU vkCmdCopyBuffer(A -> B), and concurrent host
// flush(A) on one thread and invalidate(B) on another.
//
// Per Vulkan spec, vkFlushMappedMemoryRanges and vkInvalidateMappedMemoryRanges
// have no external-synchronization requirement on VkDeviceMemory (only on
// VkDevice), and disjoint nonCoherentAtomSize-aligned ranges of one
// allocation may be flushed/invalidated concurrently from different threads.
// The buffer layout below ensures A and B never share an atom -- offsets,
// stride, and atom are printed at startup so the layout is auditable.
//
// On the affected stack the invalidate of B returns stale data when it
// races against the flush of A. Validation layers do NOT catch this --
// it is a driver cache-coherency bug, not API misuse. The REPRO_VALIDATION
// env var is provided only to demonstrate that fact.
//
// Build:  c++ -std=c++17 -O2 tiny_repro.cpp -lvulkan -o tiny_repro
// Run:    ./tiny_repro
//         REPRO_VALIDATION=1 VK_LAYER_PATH=... ./tiny_repro   (optional)
//
// Affected stack: thousands of mismatches per 10000 iterations.
// Working stack:  0/10000.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#define VK(e) do { VkResult _r = (e); if (_r) { std::fprintf(stderr, "VK %s = %d\n", #e, _r); std::exit(2); } } while (0)

static constexpr uint32_t kFrames = 2;
static constexpr uint32_t kIters  = 10000;
static constexpr VkDeviceSize kBufBytes = 4096;
static constexpr uint32_t kInvalidIdx = static_cast<uint32_t>(-1);

struct Frame {
    VkBuffer bufA, bufB;
    VkDeviceSize offA, offB;
    void *mapA, *mapB;
    VkCommandBuffer cmd;
    VkFence fence;
    bool busy = false;     // protected by `mu` below
    uint32_t value = 0;
};

static VkDeviceSize alignUp(VkDeviceSize x, VkDeviceSize a) { return (x + a - 1) & ~(a - 1); }

int main() {
    // Instance.
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;

    std::vector<const char*> layers;
    if (std::getenv("REPRO_VALIDATION")) {
        const char* w = "VK_LAYER_KHRONOS_validation";
        uint32_t n = 0; vkEnumerateInstanceLayerProperties(&n, nullptr);
        std::vector<VkLayerProperties> avail(n);
        vkEnumerateInstanceLayerProperties(&n, avail.data());
        for (auto& l : avail) if (!std::strcmp(l.layerName, w)) { layers.push_back(w); break; }
        std::fprintf(stderr, layers.empty() ? "validation requested but layer absent\n"
                                            : "validation enabled\n");
    }
    const char* exts[] = {
#ifdef __APPLE__
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, "VK_KHR_get_physical_device_properties2",
#endif
    };
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = (uint32_t)layers.size(); ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = sizeof(exts)/sizeof(exts[0]); ici.ppEnabledExtensionNames = exts;
#ifdef __APPLE__
    ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    VkInstance instance; VK(vkCreateInstance(&ici, nullptr, &instance));

    // Physical device: query count, then prefer a discrete GPU.
    uint32_t pdN = 0; VK(vkEnumeratePhysicalDevices(instance, &pdN, nullptr));
    if (pdN == 0) { std::fprintf(stderr, "no Vulkan physical devices\n"); return 3; }
    std::vector<VkPhysicalDevice> pds(pdN);
    VK(vkEnumeratePhysicalDevices(instance, &pdN, pds.data()));
    VkPhysicalDevice pd = pds[0];
    for (auto cand : pds) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(cand, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = cand; break; }
    }
    VkPhysicalDeviceProperties pdp; vkGetPhysicalDeviceProperties(pd, &pdp);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t qfN; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfN);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, qfs.data());
    uint32_t qfi = kInvalidIdx;
    // vkCmdCopyBuffer needs TRANSFER (or COMPUTE/GRAPHICS, which imply it).
    for (uint32_t i = 0; i < qfN; ++i)
        if (qfs[i].queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) { qfi = i; break; }
    if (qfi == kInvalidIdx) { std::fprintf(stderr, "no transfer-capable queue family\n"); return 3; }

    // Device.
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
    dci.enabledExtensionCount = sizeof(devExts)/sizeof(devExts[0]); dci.ppEnabledExtensionNames = devExts;
    VkDevice device; VK(vkCreateDevice(pd, &dci, nullptr, &device));
    VkQueue queue; vkGetDeviceQueue(device, qfi, 0, &queue);

    // Memory type: DEVICE_LOCAL+HOST_VISIBLE+HOST_CACHED, NOT HOST_COHERENT.
    uint32_t mt = kInvalidIdx;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)  &&
           !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
    }
    if (mt == kInvalidIdx) { std::fprintf(stderr, "no non-coherent device-local host-cached memory type\n"); return 3; }

    // Buffers + ONE shared VkDeviceMemory containing all frames' A and B.
    auto mkbuf = [&](VkBufferUsageFlags usage) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = kBufBytes;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer b; VK(vkCreateBuffer(device, &bci, nullptr, &b)); return b;
    };
    Frame frames[kFrames];
    for (auto& fr : frames) {
        fr.bufA = mkbuf(VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        fr.bufB = mkbuf(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    }

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(device, frames[0].bufA, &req);
    const VkDeviceSize atom   = pdp.limits.nonCoherentAtomSize;
    const VkDeviceSize stride = alignUp(req.size, std::max<VkDeviceSize>(req.alignment, atom));

    VkDeviceSize cursor = 0;
    for (auto& fr : frames) { fr.offA = cursor; cursor += stride; fr.offB = cursor; cursor += stride; }

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = alignUp(cursor, atom); mai.memoryTypeIndex = mt;
    VkDeviceMemory mem; VK(vkAllocateMemory(device, &mai, nullptr, &mem));
    void* base; VK(vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &base));
    for (auto& fr : frames) {
        VK(vkBindBufferMemory(device, fr.bufA, mem, fr.offA));
        VK(vkBindBufferMemory(device, fr.bufB, mem, fr.offB));
        fr.mapA = static_cast<char*>(base) + fr.offA;
        fr.mapB = static_cast<char*>(base) + fr.offB;
    }

    // Print atom layout so reviewers can confirm A and B never share an atom.
    std::printf("device: %s\n", pdp.deviceName);
    std::printf("nonCoherentAtomSize=%llu  bufSize=%llu  bufAlignment=%llu  stride=%llu  alloc=%llu\n",
                (unsigned long long)atom, (unsigned long long)kBufBytes,
                (unsigned long long)req.alignment, (unsigned long long)stride,
                (unsigned long long)mai.allocationSize);
    for (uint32_t i = 0; i < kFrames; ++i)
        std::printf("  frame %u: offA=%llu offB=%llu  (atom-aligned: A=%s B=%s)\n", i,
                    (unsigned long long)frames[i].offA, (unsigned long long)frames[i].offB,
                    (frames[i].offA % atom == 0) ? "yes" : "NO",
                    (frames[i].offB % atom == 0) ? "yes" : "NO");

    // Per-frame command buffer + fence.
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; VK(vkCreateCommandPool(device, &cpci, nullptr, &cpool));
    for (auto& fr : frames) {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VK(vkAllocateCommandBuffers(device, &cbai, &fr.cmd));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK(vkCreateFence(device, &fci, nullptr, &fr.fence));
    }

    // All cross-thread state (jobs queue, per-frame `busy`, stop) is owned
    // by `mu`. Notifications happen while holding the mutex so the producer
    // can't sleep through a state change it was about to observe.
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
            r.memory = mem; r.offset = fr.offB; r.size = stride;
            VK(vkInvalidateMappedMemoryRanges(device, 1, &r));    // <-- bug surface
            const uint32_t* B = static_cast<const uint32_t*>(fr.mapB);
            uint32_t expect = fr.value;
            uint32_t local_bad = 0;
            for (uint32_t i = 0; i < kBufBytes / 4; ++i) {
                if (B[i] != expect) { local_bad = 1; break; }
            }
            { std::lock_guard<std::mutex> g(mu); fr.busy = false; bad += local_bad; }
            cv.notify_all();
        }
    });

    // Main loop.
    for (uint32_t it = 0; it < kIters; ++it) {
        uint32_t idx = it % kFrames; Frame& fr = frames[idx];
        { std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&]{ return !fr.busy; }); }

        // Host writes A; GPU copies A -> B; host (on reader thread) reads B.
        fr.value = 0xC0DE0000u | it;
        uint32_t* A = static_cast<uint32_t*>(fr.mapA);
        for (uint32_t i = 0; i < kBufBytes / 4; ++i) A[i] = fr.value;

        VK(vkResetCommandBuffer(fr.cmd, 0));
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK(vkBeginCommandBuffer(fr.cmd, &cbbi));
        VkBufferCopy region{0, 0, kBufBytes};
        vkCmdCopyBuffer(fr.cmd, fr.bufA, fr.bufB, 1, &region);
        // Post-barrier amplifies the race; the bug also reproduces (less often) without it.
        VkBufferMemoryBarrier post{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; post.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        post.srcQueueFamilyIndex = post.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post.buffer = fr.bufB; post.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(fr.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &post, 0, nullptr);
        VK(vkEndCommandBuffer(fr.cmd));

        // Flush A's range -- the call that races with the reader's invalidate.
        VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        r.memory = mem; r.offset = fr.offA; r.size = stride;
        VK(vkFlushMappedMemoryRanges(device, 1, &r));

        VK(vkResetFences(device, 1, &fr.fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &fr.cmd;
        VK(vkQueueSubmit(queue, 1, &si, fr.fence));

        { std::lock_guard<std::mutex> g(mu); fr.busy = true; jobs.push(idx); }
        cv.notify_one();
    }

    { std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&]{
          if (!jobs.empty()) return false;
          for (auto& fr : frames) if (fr.busy) return false;
          return true; });
      stop = true; }
    cv.notify_all();
    reader.join();

    std::printf("RESULT: %u / %u iterations bad\n", bad, kIters);
    return bad == 0 ? 0 : 1;
}
