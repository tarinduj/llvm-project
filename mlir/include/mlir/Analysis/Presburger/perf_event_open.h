#define _GNU_SOURCE
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "assert.h"

#include <inttypes.h>

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                    group_fd, flags);
    return ret;
}

#define FD_ARRAY_LENGTH  30
int fd[FD_ARRAY_LENGTH];
__u64 event_id[FD_ARRAY_LENGTH];
unsigned fd_count = 0;

int get_fd(int group_fd, uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = type;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    // Don't count hypervisor events.
    pe.exclude_hv = 1;
    pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return perf_event_open(&pe, 0, -1, group_fd, 0);
}

void set_fd(__u32 type, __u64 config) {
  fd[fd_count] = get_fd(fd_count == 0 ? -1 : fd[0], type, config);
  ioctl(fd[fd_count], PERF_EVENT_IOC_ID, &event_id[fd_count]);
  // fprintf(stderr, "%d: id(%d) = %lld\n", fd_count, fd[fd_count], event_id[fd_count]);
  fd_count++;
}

void reset_and_enable(int fd) {
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

long long get_count(int fd) {
  long long count;
  read(fd, &count, sizeof(long long));
  return count;
}

void init_perf_fds(unsigned mode) {
  for (unsigned i = 0; i < FD_ARRAY_LENGTH; ++i)
    fd[i] = -1;

  if (mode == 0) {
    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));

    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));

    set_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    set_fd(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
  } else if (mode == 1) {
    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16));
    set_fd(PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16));
  }

  for (unsigned i = 0; i < fd_count; ++i) {
    if (fd[i] == -1) {
      perror("error: perf_event_open");
      exit(1);
    }
  }

  assert(fd_count <= FD_ARRAY_LENGTH);
}

void reset_and_enable_all() {
  assert(fd_count != 0 && "fds not initialized!");
  ioctl(fd[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  // for (unsigned i = 0; i < fd_count; ++i)
  //   reset_and_enable(fd[i]);
}

typedef struct read_format {
  uint64_t nr;
  uint64_t time_enabled;
  uint64_t time_running;
  struct {
    uint64_t value;
    uint64_t id;
  } values[FD_ARRAY_LENGTH];
} read_format;

void disable_all_and_print_counts(void (*print)(uint64_t)) {
  ioctl(fd[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  read_format data;
  unsigned read_bytes = read(fd[0], &data, sizeof(data));

  assert(read_bytes == 24 + 16*fd_count);
  assert(data.nr == fd_count);
  print(data.time_enabled);
  print(data.time_running);
  for (unsigned i = 0; i < fd_count; ++i) {
    for (unsigned j = 0; j < data.nr; ++j) {
      if (data.values[j].id == event_id[i]) {
        print(data.values[j].value);
        break;
      }
    }
  }
  assert(fd_count != 0 && "fds not initialized!");
}

