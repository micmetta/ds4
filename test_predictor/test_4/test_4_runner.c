/* TEST_4: controlled oracle replay plus SSD -> pinned-RAM prefetch.
 * Run A writes the reference, Run B is the replay baseline, and Run C uses
 * exactly the same replay while enabling the separate pinned prefetch bank. */

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
        const int rc = i == 1 ? snprintf(out, n, "test_predictor/test_4/output") :
            snprintf(out, n, "test_predictor/test_4/output_%u", i);
        if (rc < 0 || (size_t)rc >= n) return false;
        if (mkdir(out, 0755) == 0) return true;
        if (errno != EEXIST) return false;
    }
    return false;
}

static int run_ds4(const char *model, const char *stdout_path, const char *stderr_path,
                   const char *oracle_write, const char *oracle_replay,
                   const char *routing_trace, const char *prefetch_trace,
                   bool prefetch) {
    const pid_t pid = fork();
    if (pid < 0) return 127;
    if (pid == 0) {
        const int out = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0 ||
            unsetenv("DS4_ROUTING_ORACLE_WRITE_PATH") ||
            unsetenv("DS4_ROUTING_ORACLE_REPLAY_PATH") ||
            unsetenv("DS4_ROUTING_TRACE_PATH") ||
            unsetenv("DS4_TEST_4_TRACE_PATH") ||
            unsetenv("DS4_TEST_4_PREFETCH") ||
            setenv("DS4_CUDA_MMQ", "0", 1) ||
            (oracle_write && setenv("DS4_ROUTING_ORACLE_WRITE_PATH", oracle_write, 1)) ||
            (oracle_replay && setenv("DS4_ROUTING_ORACLE_REPLAY_PATH", oracle_replay, 1)) ||
            (routing_trace && setenv("DS4_ROUTING_TRACE_PATH", routing_trace, 1)) ||
            (prefetch_trace && setenv("DS4_TEST_4_TRACE_PATH", prefetch_trace, 1)) ||
            (prefetch && setenv("DS4_TEST_4_PREFETCH", "1", 1))) _exit(127);
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

static unsigned long count_event(const char *path, const char *event) {
    FILE *fp = fopen(path, "r");
    char line[2048], needle[96];
    unsigned long count = 0;
    if (!fp) return 0;
    snprintf(needle, sizeof(needle), "\"event\":\"%s\"", event);
    while (fgets(line, sizeof(line), fp)) if (strstr(line, needle)) ++count;
    fclose(fp);
    return count;
}

static double generation_tps(const char *path) {
    FILE *fp = fopen(path, "r");
    char line[4096];
    double tps = 0.0;
    if (!fp) return 0.0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = strstr(line, "generation:");
        if (p) (void)sscanf(p, "generation: %lf t/s", &tps);
    }
    fclose(fp);
    return tps;
}

static unsigned test4_prefetch_layer_count(void) {
    const char *value = getenv("DS4_TEST_4_PREFETCH_LAYERS");
    if (!value || !value[0]) return 5u;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') return 5u;
    return parsed > 43u ? 43u : (unsigned)parsed;
}

static bool write_summary(const char *path, int a, int b, int c,
                          const char *b_stderr, const char *c_stderr,
                          const char *b_trace, const char *c_trace) {
    time_t now = time(NULL); struct tm utc; char stamp[32];
    if (!gmtime_r(&now, &utc) || !strftime(stamp, sizeof(stamp),
                                           "%Y-%m-%dT%H:%M:%SZ", &utc)) return false;
    // DS4 writes the "prefill / generation" performance line to stderr.
    const double b_tps = generation_tps(b_stderr);
    const double c_tps = generation_tps(c_stderr);
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    const int rc = fprintf(fp,
        "{\n  \"test\": \"TEST_4 oracle pinned-RAM prefetch\",\n"
        "  \"timestamp_utc\": \"%s\",\n"
        "  \"run_a_exit_code\": %d,\n  \"run_b_exit_code\": %d,\n"
        "  \"run_c_exit_code\": %d,\n"
        "  \"run_b_generation_tps\": %.6f,\n"
        "  \"run_c_generation_tps\": %.6f,\n"
        "  \"run_c_prefetch_layers\": %u,\n"
        "  \"run_b_prefetch_hits\": %lu,\n"
        "  \"run_c_prefetch_hits\": %lu,\n"
        "  \"run_c_prefetch_miss_demand\": %lu,\n"
        "  \"run_c_prefetch_enqueues\": %lu\n}\n",
        stamp, a, b, c, b_tps, c_tps, test4_prefetch_layer_count(),
        count_event(b_trace, "prefetch_hit"),
        count_event(c_trace, "prefetch_hit"),
        count_event(c_trace, "prefetch_miss_demand"),
        count_event(c_trace, "prefetch_enqueue"));
    return rc >= 0 && fclose(fp) == 0;
}

int main(void) {
    const char *model = getenv("DS4_MODEL");
    if (access("./ds4", X_OK) || !model || !model[0] || access(model, R_OK)) {
        fprintf(stderr, "TEST_4: run from repository root with DS4_MODEL set to the GGUF file\n");
        return 2;
    }
    char dir[PATH_MAX], oracle[PATH_MAX], summary[PATH_MAX];
    char aout[PATH_MAX], aerr[PATH_MAX], bout[PATH_MAX], berr[PATH_MAX], cout[PATH_MAX], cerr[PATH_MAX];
    char aroute[PATH_MAX], broute[PATH_MAX], croute[PATH_MAX];
    char atrace[PATH_MAX], btrace[PATH_MAX], ctrace[PATH_MAX];
    if (!make_output(dir, sizeof(dir)) ||
        !join(oracle, sizeof(oracle), dir, "routing_oracle.bin") ||
        !join(summary, sizeof(summary), dir, "summary.json") ||
        !join(aout, sizeof(aout), dir, "run_A.stdout.txt") || !join(aerr, sizeof(aerr), dir, "run_A.stderr.txt") ||
        !join(bout, sizeof(bout), dir, "run_B.stdout.txt") || !join(berr, sizeof(berr), dir, "run_B.stderr.txt") ||
        !join(cout, sizeof(cout), dir, "run_C.stdout.txt") || !join(cerr, sizeof(cerr), dir, "run_C.stderr.txt") ||
        !join(aroute, sizeof(aroute), dir, "run_A.routing.jsonl") ||
        !join(broute, sizeof(broute), dir, "run_B.routing.jsonl") ||
        !join(croute, sizeof(croute), dir, "run_C.routing.jsonl") ||
        !join(atrace, sizeof(atrace), dir, "run_A.prefetch.jsonl") ||
        !join(btrace, sizeof(btrace), dir, "run_B.prefetch.jsonl") ||
        !join(ctrace, sizeof(ctrace), dir, "run_C.prefetch.jsonl")) return 2;

    printf("TEST_4: writing results to %s\n", dir);
    printf("TEST_4: starting Run A to create the complete routing oracle\n");
    const int a = run_ds4(model, aout, aerr, oracle, NULL, aroute, atrace, false);
    if (a) { fprintf(stderr, "TEST_4: Run A failed\n"); return a; }
    printf("TEST_4: starting Run B with complete oracle replay and no prefetch\n");
    const int b = run_ds4(model, bout, berr, NULL, oracle, broute, btrace, false);
    if (b) { fprintf(stderr, "TEST_4: Run B failed\n"); return b; }
    printf("TEST_4: starting Run C with complete oracle replay and pinned-RAM prefetch\n");
    const unsigned prefetch_layers = test4_prefetch_layer_count();
    if (prefetch_layers == 0) {
        printf("TEST_4: Run C prefetch window disabled (0 layers)\n");
    } else {
        printf("TEST_4: Run C prefetch window: layers 0..%u\n", prefetch_layers - 1u);
    }
    const int c = run_ds4(model, cout, cerr, NULL, oracle, croute, ctrace, true);
    if (c) { fprintf(stderr, "TEST_4: Run C failed\n"); return c; }
    if (!write_summary(summary, a, b, c, berr, cerr, btrace, ctrace)) return 2;
    printf("TEST_4: Run A/B/C traces and summary written to %s\n", dir);
    return 0;
}
