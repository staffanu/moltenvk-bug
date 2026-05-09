// Tiniest repro of the MoltenVK / AMD / macOS cache-coherency bug.
// No compute, no shader, no descriptors. Just two buffers in one
// VkDeviceMemory, vkCmdFillBuffer to write one of them on the GPU, and
// concurrent host flush(A) on one thread and invalidate(B) on another.
//
// Per Vulkan spec, vkFlushMappedMemoryRanges and vkInvalidateMappedMemoryRanges
// have no external-synchronization requirement, and disjoint
// nonCoherentAtomSize-aligned ranges of the same VkDeviceMemory may be
// flushed/invalidated concurrently from multiple threads. On the affected
// stack, the invalidate returns stale data when it races against the flush.
//
// Build:  c++ -std=c++17 -O2 tiny_repro.cpp -lvulkan -o tiny_repro
// Run:    ./tiny_repro
//         REPRO_VALIDATION=1 VK_LAYER_PATH=... ./tiny_repro   (optional)
//
// Affected stack: thousands of mismatches per 10000 iterations.
// Working stack:  0/10000.

#include <vulkan/vulkan.h>

#include <atomic>
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

struct Frame {
    VkBuffer bufA, bufB;
    VkDeviceSize offA, offB;
    void *mapA, *mapB;
    VkCommandBuffer cmd;
    VkFence fence;
    enum class State { Ready, Submitted };
    std::atomic<State> state{State::Ready};
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

    // Physical device + queue family.
    uint32_t n = 1; VkPhysicalDevice pd; VK(vkEnumeratePhysicalDevices(instance, &n, &pd));
    VkPhysicalDeviceProperties pdp; vkGetPhysicalDeviceProperties(pd, &pdp);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t qfN; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfN);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfN, qfs.data());
    uint32_t qfi = ~0u;
    // vkCmdFillBuffer needs TRANSFER (or COMPUTE/GRAPHICS) — every queue family supports it implicitly.
    for (uint32_t i = 0; i < qfN; ++i)
        if (qfs[i].queueFlags & (VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) { qfi = i; break; }

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
    uint32_t mt = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)  &&
           !(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { mt = i; break; }
    }
    if (mt == ~0u) { std::fprintf(stderr, "no non-coherent device-local host-cached memory type\n"); return 3; }

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

    // Reader thread: pulls frame indices off a queue, invalidates B, compares.
    std::queue<uint32_t> jobs; std::mutex mu; std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> bad{0};
    std::thread reader([&]() {
        while (true) {
            uint32_t idx;
            { std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&]{ return stop.load() || !jobs.empty(); });
              if (jobs.empty() && stop.load()) return;
              idx = jobs.front(); jobs.pop(); }
            Frame& fr = frames[idx];
            VK(vkWaitForFences(device, 1, &fr.fence, VK_TRUE, UINT64_MAX));
            VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            r.memory = mem; r.offset = fr.offB; r.size = stride;
            VK(vkInvalidateMappedMemoryRanges(device, 1, &r));    // <-- bug surface
            const uint32_t* B = static_cast<const uint32_t*>(fr.mapB);
            uint32_t expect = fr.value;
            for (uint32_t i = 0; i < kBufBytes / 4; ++i) {
                if (B[i] != expect) { bad.fetch_add(1); break; }
            }
            fr.state.store(Frame::State::Ready, std::memory_order_release);
            cv.notify_all();
        }
    });

    // Main loop.
    for (uint32_t it = 0; it < kIters; ++it) {
        uint32_t idx = it % kFrames; Frame& fr = frames[idx];
        { std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&]{ return fr.state.load(std::memory_order_acquire) == Frame::State::Ready; }); }

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

        fr.state.store(Frame::State::Submitted, std::memory_order_release);
        { std::lock_guard<std::mutex> g(mu); jobs.push(idx); } cv.notify_one();
    }

    { std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&]{
          if (!jobs.empty()) return false;
          for (auto& fr : frames) if (fr.state.load() != Frame::State::Ready) return false;
          return true; });
      stop.store(true); cv.notify_all(); }
    reader.join();

    std::printf("device: %s  nonCoherentAtomSize=%llu\n", pdp.deviceName, (unsigned long long)atom);
    std::printf("RESULT: %u / %u iterations bad\n", bad.load(), kIters);
    return bad.load() == 0 ? 0 : 1;
}
