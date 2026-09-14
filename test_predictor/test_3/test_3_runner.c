/* TEST_3 sub-test: measure VRAM around the real demand load on a forced path. */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static bool join(char *out, size_t n, const char *dir, const char *name) {
    const int rc = snprintf(out, n, "%s/%s", dir, name);
    return rc >= 0 && (size_t)rc < n;
}

static bool make_output(char *out, size_t n) {
    for (unsigned i = 1; i < 1000000; ++i) {
        const int rc = i == 1 ? snprintf(out, n, "test_predictor/test_3/output") :
            snprintf(out, n, "test_predictor/test_3/output_%u", i);
        if (rc < 0 || (size_t)rc >= n) return false;
        if (mkdir(out, 0755) == 0) return true;
        if (errno != EEXIST) return false;
    }
    return false;
}

static int run_ds4(const char *model, const char *stdout_path, const char *stderr_path,
                   const char *oracle_write, const char *oracle_replay,
                   const char *vram_ram_trace) {
    const pid_t pid = fork();
    if (pid < 0) return 127;
    if (pid == 0) {
        const int out = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0 || setenv("DS4_CUDA_MMQ", "0", 1) ||
            (oracle_write && setenv("DS4_ROUTING_ORACLE_WRITE_PATH", oracle_write, 1)) ||
            (oracle_replay && setenv("DS4_ROUTING_ORACLE_REPLAY_PATH", oracle_replay, 1)) ||
            (vram_ram_trace && setenv("DS4_TEST_3_VRAM_TRACE_PATH", vram_ram_trace, 1))) _exit(127);
        close(out); close(err);
        execl("./ds4", "./ds4", "--cuda", "--ssd-streaming",
              "--ssd-streaming-cache-experts", "8GB", "-c", "1024", "-n", "64",
              "--nothink", "--temp", "0", "-p",
              "Explain what a binary search is in one short paragraph.", "-m", model,
              (char *)NULL);
        _exit(127);
    }
    int status = 0;
    return waitpid(pid, &status, 0) < 0 ? 127 :
        (WIFEXITED(status) ? WEXITSTATUS(status) : 128);
}

static unsigned long count_stage(const char *trace_path, const char *stage) {
    FILE *fp = fopen(trace_path, "r");
    char line[2048], needle[96];
    unsigned long count = 0;
    snprintf(needle, sizeof(needle), "\"stage\":\"%s\"", stage);
    if (!fp) return 0;
    while (fgets(line, sizeof(line), fp)) if (strstr(line, needle)) ++count;
    fclose(fp);
    return count;
}

static bool write_summary(const char *path, const char *trace, int a, int b) {
    time_t now = time(NULL); struct tm utc; char stamp[32];
    if (!gmtime_r(&now, &utc) || !strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &utc)) return false;
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    const int rc = fprintf(fp,
        "{\n  \"test\": \"TEST_3 VRAM and RAM capacity sub-test\",\n"
        "  \"timestamp_utc\": \"%s\",\n  \"run_a_exit_code\": %d,\n"
        "  \"run_b_exit_code\": %d,\n  \"before_demand_load_events\": %lu,\n"
        "  \"after_demand_load_events\": %lu,\n  \"after_moe_queued_events\": %lu,\n"
        "  \"trace\": \"vram_ram_trace.jsonl\"\n}\n", stamp, a, b,
        count_stage(trace, "before_demand_load"), count_stage(trace, "after_demand_load"),
        count_stage(trace, "after_moe_queued"));
    return rc >= 0 && fclose(fp) == 0;
}

int main(void) {
    const char *model = getenv("DS4_MODEL");
    if (access("./ds4", X_OK) || !model || !model[0] || access(model, R_OK)) {
        fprintf(stderr, "TEST_3: run from repository root with DS4_MODEL set to the GGUF file\n");
        return 2;
    }
    char dir[PATH_MAX], oracle[PATH_MAX], trace[PATH_MAX], summary[PATH_MAX];
    char aout[PATH_MAX], aerr[PATH_MAX], bout[PATH_MAX], berr[PATH_MAX];
    if (!make_output(dir, sizeof(dir)) || !join(oracle, sizeof(oracle), dir, "routing_oracle.bin") ||
        !join(trace, sizeof(trace), dir, "vram_ram_trace.jsonl") || !join(summary, sizeof(summary), dir, "summary.json") ||
        !join(aout, sizeof(aout), dir, "run_A.stdout.txt") || !join(aerr, sizeof(aerr), dir, "run_A.stderr.txt") ||
        !join(bout, sizeof(bout), dir, "run_B.stdout.txt") || !join(berr, sizeof(berr), dir, "run_B.stderr.txt")) return 2;
    printf("TEST_3: writing results to %s\n", dir);
    printf("TEST_3: starting Run A to create the routing oracle\n");
    const int a = run_ds4(model, aout, aerr, oracle, NULL, NULL);
    if (a) { fprintf(stderr, "TEST_3: Run A failed\n"); return a; }
    printf("TEST_3: starting Run B with oracle replay and VRAM/RAM probes\n");
    const int b = run_ds4(model, bout, berr, NULL, oracle, trace);
    if (b) { fprintf(stderr, "TEST_3: Run B failed\n"); return b; }
    if (!write_summary(summary, trace, a, b)) return 2;
    printf("TEST_3: VRAM/RAM trace written to %s\n", trace);
    printf("TEST_3: summary written to %s\n", summary);
    return 0;
}
