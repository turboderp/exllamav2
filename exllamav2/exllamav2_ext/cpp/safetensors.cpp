#include "safetensors.h"

#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <c10/cuda/CUDAGuard.h>
#include <thread>
#include "util.h"
#include <aio.h>

#define MAX_BLOCK_SIZE (128*1024)
#define MAX_PAGES 8
#define PAGESIZE (8*1024*1024)
#define PINNED_MEMORY (MAX_PAGES * PAGESIZE)

uintptr_t safetensors_open(const char* filename, uintptr_t pinned_buffer)
{
    STFile* f = new STFile(filename, pinned_buffer);
    return reinterpret_cast<uintptr_t> (f);
}

void safetensors_close(uintptr_t handle)
{
    STFile* f = reinterpret_cast<STFile*> (handle);
    delete f;
}

uintptr_t safetensors_pinned_buffer()
{
    uintptr_t b;
    cudaMallocHost((void**)&b, PINNED_MEMORY + 2 * MAX_BLOCK_SIZE);
    TORCH_CHECK(b, "Unable to allocate pinned memory");
    return b;
}

void safetensors_free_pinned_buffer(uintptr_t b)
{
    cudaFreeHost((void*)b);
}

STFile::STFile(const char* filename, uintptr_t pinned_buffer)
{
    file_descriptor = open(filename, O_RDONLY | O_DIRECT);
    TORCH_CHECK(file_descriptor != -1, "Safetensors file I/O error");

    struct stat sb;
    auto res = fstat(file_descriptor, &sb);
    TORCH_CHECK(res != -1, "Safetensors fstat failed");
    filesize = sb.st_size;
    block_size = sb.st_blksize;

    padded_size = DIVIDE(filesize, block_size) * block_size;

    TORCH_CHECK(block_size <= MAX_BLOCK_SIZE, "Block size too large")

    char *aligned_ptr = (char *)(((uintptr_t)pinned_buffer + block_size - 1) & ~(block_size - 1));
    aligned_buffer = (void*) (aligned_ptr + MAX_BLOCK_SIZE);
}

STFile::~STFile()
{
    close(file_descriptor);
}

void STFile::fastload
(
    std::vector<torch::Tensor>& targets,
    std::vector<size_t> offsets,
    std::vector<size_t> lengths,
    size_t h_offset
)
{
    int num_targets = targets.size();
    std::vector<size_t> target_offsets(num_targets);

    size_t min = offsets[0] + h_offset;
    size_t max = min + lengths[0];
    for (int i = 0; i < num_targets; ++i)
    {
        size_t a = offsets[i] + h_offset;
        size_t b = a + lengths[i];
        if (a < min) min = a;
        if (b > max) max = b;
        target_offsets[i] = 0;
        offsets[i] += h_offset;
    }

    size_t file_min = min / block_size * block_size;
    size_t file_max = DIVIDE(max, block_size) * block_size;

//    DBGI2(file_min, file_max);

    size_t file_offset = file_min;
    size_t file_offset_read = file_offset;
    size_t file_remaining = file_max - file_min;

    while (file_remaining)
    {
        aiocb aiocb_list[MAX_PAGES];
        size_t chunk_begin[MAX_PAGES];
        size_t chunk_end[MAX_PAGES];

        size_t buffer_offset = 0;

        cudaDeviceSynchronize();

        for (int i = 0; i < MAX_PAGES && file_remaining; ++i)
        {
            size_t chunk_size = PAGESIZE;
            if (chunk_size > file_remaining) chunk_size = file_remaining;

            memset(&aiocb_list[i], 0, sizeof(aiocb));
            aiocb_list[i].aio_fildes = file_descriptor;
            aiocb_list[i].aio_buf = ((char*) aligned_buffer) + buffer_offset;
            aiocb_list[i].aio_nbytes = chunk_size;
            aiocb_list[i].aio_offset = file_offset;
            aio_read(&aiocb_list[i]);

//            printf("--- enqueued:\n");
//            DBGI3(buffer_offset, chunk_size, file_offset);

            chunk_begin[i] = file_offset;
            chunk_end[i] = file_offset + chunk_size;

            file_offset += chunk_size;
            buffer_offset += chunk_size;
            file_remaining -= chunk_size;
        }

        int page_i = 0;
        size_t page_offset = 0;
        size_t file_offset_buf = file_offset_read;
        buffer_offset = 0;
        while (file_offset_read < file_offset)
        {
            struct aiocb *aiocb_active[1];
            aiocb_active[0] = &aiocb_list[page_i];
            aio_suspend(aiocb_active, 1, NULL);

            int err = aio_error(&aiocb_list[page_i]);
            TORCH_CHECK(err == 0, "Async read error (1)");
            ssize_t bytes_read = aio_return(&aiocb_list[page_i]);
            TORCH_CHECK(bytes_read > 0, "Async read error (2)");

            size_t a = chunk_begin[page_i];
            size_t b = chunk_end[page_i];

//            printf("--- read:\n");
//            DBGI2(a, b);
//            DBGI2(page_offset, b - a);

            for (int j = 0; j < num_targets; ++j)
            {
                size_t file_t_a = offsets[j];
                size_t file_t_b = offsets[j] + lengths[j];
                if (file_t_a < a) file_t_a = a;
                if (file_t_b > b) file_t_b = b;
                ssize_t copy_len = file_t_b - file_t_a;
                if (copy_len <= 0) continue;

                size_t buffer_offset = file_t_a - file_offset_buf;

//                printf("--- tensor chunk:\n");
//                DBGI(j);
//                DBGI3(file_t_a, file_t_b, copy_len);
//                DBGI(buffer_offset);

                char* src = ((char*) aligned_buffer) + buffer_offset;
                char* dst = ((char*) targets[j].data_ptr()) + target_offsets[j];
                cudaMemcpyAsync(dst, src, copy_len, cudaMemcpyHostToDevice);

                offsets[j] += copy_len;
                lengths[j] -= copy_len;
                target_offsets[j] += copy_len;

//                DBGI3(offsets[j], lengths[j], target_offsets[j]);
            }

            page_offset += b - a;
            file_offset_read = b;
            page_i++;
        }
    }

}

void safetensors_fastload
(
    uintptr_t handle,
    std::vector<torch::Tensor>& targets,
    std::vector<size_t> offsets,
    std::vector<size_t> lengths,
    size_t h_offset
)
{
    STFile* f = reinterpret_cast<STFile*> (handle);
    f->fastload(targets, offsets, lengths, h_offset);
}

