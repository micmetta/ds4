/* TEST_2: one CUDA + SSD-streaming run with selected-expert load tracing. */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static bool make_output_dir(char *path, size_t path_size) {
    const char *root = "test_predictor/test_2/output";
    for (unsigned index = 1; index < 1000000; index++) {
        const int written = index == 1 ? snprintf(path, path_size, "%s", root) :
            snprintf(path, path_size, "%s_%u", root, index);
        if (written < 0 || (size_t)written >= path_size) return false;
        if (mkdir(path, 0755) == 0) return true;
        if (errno != EEXIST) {
            fprintf(stderr, "TEST_2: cannot create %s: %s\n", path, strerror(errno));
            return false;
        }
    }
    return false;
}

static bool join_path(char *dst, size_t dst_size, const char *dir, const char *name) {
    const int written = snprintf(dst, dst_size, "%s/%s", dir, name);
    return written >= 0 && (size_t)written < dst_size;
}

static bool write_summary(const char *path, const char *trace_path, int child_status) {
    FILE *trace = fopen(trace_path, "r");
    if (!trace) {
        fprintf(stderr, "TEST_2: cannot read trace %s: %s\n", trace_path, strerror(errno));
        return false;
    }
    unsigned long events = 0;
    unsigned long direct_io_events = 0;
    char line[2048];
    while (fgets(line, sizeof(line), trace)) {
        events++;
        if (strstr(line, "\"direct_io\":true")) direct_io_events++;
    }
    const bool trace_ok = !ferror(trace) && fclose(trace) == 0;
    if (!trace_ok) return false;

    const time_t now = time(NULL);
    struct tm utc;
    if (!gmtime_r(&now, &utc)) return false;
    char timestamp[32];
    if (strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc) == 0) return false;

    FILE *summary = fopen(path, "w");
    if (!summary) return false;
    const int rc = fprintf(summary,
        "{\n"
        "  \"test\": \"TEST_2 SSD streaming trace\",\n"
        "  \"timestamp_utc\": \"%s\",\n"
        "  \"ds4_exit_code\": %d,\n"
        "  \"selected_load_events\": %lu,\n"
        "  \"direct_io_events\": %lu,\n"
        "  \"trace\": \"ssd_trace.jsonl\"\n"
        "}\n",
        timestamp, child_status, events, direct_io_events);
    return rc >= 0 && fclose(summary) == 0;
}

int main(void) {
    if (access("./ds4", X_OK) != 0) {
        fprintf(stderr, "TEST_2: run from repository root after building CUDA\n");
        return 2;
    }
    const char *model = getenv("DS4_MODEL");
    if (!model || !model[0] || access(model, R_OK) != 0) {
        fprintf(stderr, "TEST_2: set DS4_MODEL to a readable GGUF path\n");
        return 2;
    }

    char output_dir[PATH_MAX];
    char trace_path[PATH_MAX];
    char stdout_path[PATH_MAX];
    char stderr_path[PATH_MAX];
    char summary_path[PATH_MAX];
    if (!make_output_dir(output_dir, sizeof(output_dir)) ||
        !join_path(trace_path, sizeof(trace_path), output_dir, "ssd_trace.jsonl") ||
        !join_path(stdout_path, sizeof(stdout_path), output_dir, "run.stdout.txt") ||
        !join_path(stderr_path, sizeof(stderr_path), output_dir, "run.stderr.txt") ||
        !join_path(summary_path, sizeof(summary_path), output_dir, "summary.json")) {
        fprintf(stderr, "TEST_2: cannot prepare output paths\n");
        return 2;
    }
    printf("TEST_2: writing results to %s\n", output_dir);

    const pid_t child = fork();
    if (child < 0) {
        perror("TEST_2: fork");
        return 2;
    }
    if (child == 0) {
        const int out_fd = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err_fd = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out_fd < 0 || err_fd < 0 || dup2(out_fd, STDOUT_FILENO) < 0 ||
            dup2(err_fd, STDERR_FILENO) < 0 ||
            setenv("DS4_SSD_TRACE_PATH", trace_path, 1) != 0 ||
            setenv("DS4_CUDA_MMQ", "0", 1) != 0) {
            perror("TEST_2: child setup");
            _exit(127);
        }
        close(out_fd);
        close(err_fd);
        execl("./ds4", "./ds4",
              "--cuda", "--ssd-streaming", "--ssd-streaming-cache-experts", "8GB",
              "-c", "1024", "-n", "64", "--nothink", "--temp", "0",
              "-p", "Explain what a binary search is in one short paragraph.",
              "-m", model, (char *)NULL);
        perror("TEST_2: exec ds4");
        _exit(127);
    }

    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        perror("TEST_2: waitpid");
        return 2;
    }
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
    if (exit_code != 0) {
        fprintf(stderr, "TEST_2: DS4 run failed (exit=%d); see %s\n", exit_code, stderr_path);
        return exit_code;
    }
    if (!write_summary(summary_path, trace_path, exit_code)) {
        fprintf(stderr, "TEST_2: cannot write summary\n");
        return 2;
    }
    printf("TEST_2: trace written to %s\n", trace_path);
    printf("TEST_2: summary written to %s\n", summary_path);
    return 0;
}
