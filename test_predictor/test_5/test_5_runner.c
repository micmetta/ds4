/* TEST_5: complete oracle replay with a separate prefetch storage source.
 *
 * Run A creates the reference. Run B is the replay baseline. Run C uses the
 * same replay and the TEST_4 pinned bank, but its worker reads a byte-identical
 * GGUF from DS4_TEST_5_PREFETCH_MODEL. Demand loads still use DS4_MODEL. */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* TEST_6 reuses this small A/B/C runner with a different output directory and
 * public scheduling variables.  Keeping the process orchestration in one
 * file avoids two copies drifting apart. */
#ifndef DS4_TEST_RUNNER_LABEL
#define DS4_TEST_RUNNER_LABEL "TEST_5"
#endif
#ifndef DS4_TEST_RUNNER_NAME
#define DS4_TEST_RUNNER_NAME "TEST_5 separate-device pinned-RAM prefetch"
#endif
#ifndef DS4_TEST_RUNNER_DIR
#define DS4_TEST_RUNNER_DIR "test_predictor/test_5"
#endif
#ifndef DS4_TEST_RUNNER_INIT_DISTANCE_ENV
#define DS4_TEST_RUNNER_INIT_DISTANCE_ENV "DS4_TEST_5_PREFETCH_DISTANCE"
#endif
#ifndef DS4_TEST_RUNNER_STEP_ENV
#define DS4_TEST_RUNNER_STEP_ENV ""
#endif

static bool join(char *out, size_t n, const char *dir, const char *name) {
    const int rc = snprintf(out, n, "%s/%s", dir, name);
    return rc >= 0 && (size_t)rc < n;
}

static bool make_output(char *out, size_t n) {
    for (unsigned i = 1; i < 1000000; ++i) {
        const int rc = i == 1 ? snprintf(out, n, DS4_TEST_RUNNER_DIR "/output") :
            snprintf(out, n, DS4_TEST_RUNNER_DIR "/output_%u", i);
        if (rc < 0 || (size_t)rc >= n) return false;
        if (mkdir(out, 0755) == 0) return true;
        if (errno != EEXIST) return false;
    }
    return false;
}

static unsigned prefetch_layer_count(void) {
    const char *value = getenv("DS4_TEST_5_PREFETCH_LAYERS");
    if (!value || !value[0]) return 1u;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') return 1u;
    return parsed > 43u ? 43u : (unsigned)parsed;
}

static unsigned prefetch_init_distance(void) {
    const char *value = getenv(DS4_TEST_RUNNER_INIT_DISTANCE_ENV);
    if (!value || !value[0]) return 1u;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0u) return 1u;
    return parsed > 8u ? 8u : (unsigned)parsed;
}

static unsigned prefetch_step(void) {
    if (!DS4_TEST_RUNNER_STEP_ENV[0]) return 1u;
    const char *value = getenv(DS4_TEST_RUNNER_STEP_ENV);
    if (!value || !value[0]) return 1u;
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0u) return 1u;
    return parsed > 1024u ? 1024u : (unsigned)parsed;
}

/* These measurements are deliberately collected by the runner, after each
 * DS4 process exits.  They do not add work to the CUDA decode path. */
typedef struct {
    double wall_seconds;
    double user_cpu_seconds;
    double system_cpu_seconds;
} ds4_run_measurement;

static double seconds_from_timeval(struct timeval value) {
    return (double)value.tv_sec + (double)value.tv_usec / 1000000.0;
}

static double monotonic_seconds(void) {
    struct timespec now;
    return clock_gettime(CLOCK_MONOTONIC, &now) == 0 ?
        (double)now.tv_sec + (double)now.tv_nsec / 1000000000.0 : 0.0;
}

static int run_ds4(const char *model, const char *stdout_path, const char *stderr_path,
                   const char *oracle_write, const char *oracle_replay,
                   const char *routing_trace, const char *prefetch_trace,
                   const char *prefetch_model, unsigned layers,
                   unsigned init_distance, unsigned step,
                   ds4_run_measurement *measurement) {
    if (measurement) memset(measurement, 0, sizeof(*measurement));
    const double started_at = monotonic_seconds();
    const pid_t pid = fork();
    if (pid < 0) return 127;
    if (pid == 0) {
        const int out = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        char layer_text[16];
        char init_distance_text[16], step_text[16];
        snprintf(layer_text, sizeof(layer_text), "%u", layers);
        snprintf(init_distance_text, sizeof(init_distance_text), "%u", init_distance);
        snprintf(step_text, sizeof(step_text), "%u", step);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0 ||
            unsetenv("DS4_ROUTING_ORACLE_WRITE_PATH") ||
            unsetenv("DS4_ROUTING_ORACLE_REPLAY_PATH") ||
            unsetenv("DS4_ROUTING_TRACE_PATH") ||
            unsetenv("DS4_TEST_4_TRACE_PATH") ||
            unsetenv("DS4_TEST_4_PREFETCH") ||
            unsetenv("DS4_TEST_4_PREFETCH_LAYERS") ||
            unsetenv("DS4_TEST_4_PREFETCH_DISTANCE") ||
            unsetenv("DS4_TEST_4_PREFETCH_INIT_DISTANCE") ||
            unsetenv("DS4_TEST_4_PREFETCH_STEP") ||
            unsetenv("DS4_TEST_5_PREFETCH_MODEL") ||
            setenv("DS4_CUDA_MMQ", "0", 1) ||
            (oracle_write && setenv("DS4_ROUTING_ORACLE_WRITE_PATH", oracle_write, 1)) ||
            (oracle_replay && setenv("DS4_ROUTING_ORACLE_REPLAY_PATH", oracle_replay, 1)) ||
            (routing_trace && setenv("DS4_ROUTING_TRACE_PATH", routing_trace, 1)) ||
            (prefetch_trace && setenv("DS4_TEST_4_TRACE_PATH", prefetch_trace, 1)) ||
            (prefetch_model && setenv("DS4_TEST_4_PREFETCH", "1", 1)) ||
            (prefetch_model && setenv("DS4_TEST_4_PREFETCH_LAYERS", layer_text, 1)) ||
            (prefetch_model && setenv("DS4_TEST_4_PREFETCH_INIT_DISTANCE", init_distance_text, 1)) ||
            (prefetch_model && setenv("DS4_TEST_4_PREFETCH_STEP", step_text, 1)) ||
            (prefetch_model && setenv("DS4_TEST_5_PREFETCH_MODEL", prefetch_model, 1))) {
            _exit(127);
        }
        close(out); close(err);
        execl("./ds4", "./ds4", "--cuda", "--ssd-streaming",
              "--ssd-streaming-cache-experts", "8GB", "-c", "1024", "-n", "64",
              "--nothink", "--temp", "0", "-p",
              "Explain what a binary search is in one short paragraph.", "-m", model,
              (char *)NULL);
        _exit(127);
    }
    int status = 0;
    struct rusage usage;
    memset(&usage, 0, sizeof(usage));
    if (wait4(pid, &status, 0, &usage) < 0) return 127;
    if (measurement) {
        const double ended_at = monotonic_seconds();
        measurement->wall_seconds = ended_at > started_at ? ended_at - started_at : 0.0;
        measurement->user_cpu_seconds = seconds_from_timeval(usage.ru_utime);
        measurement->system_cpu_seconds = seconds_from_timeval(usage.ru_stime);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
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

/* Summary of the existing JSONL fields.  The parser stays intentionally
 * small: TEST_4/5/6 emit one compact object per line with fixed numeric keys. */
typedef struct {
    unsigned long demand_events;
    unsigned long hit_events;
    unsigned long enqueue_events;
    unsigned long worker_queued_events;
    unsigned long worker_started_events;
    unsigned long worker_ready_events;
    unsigned long worker_failed_events;
    unsigned long bank_full_events;
    unsigned long lookup_loading_events;
    unsigned long lookup_absent_events;
    unsigned long lookup_failed_events;
    unsigned long lookup_expert_mismatch_events;
    unsigned long stale_discarded_events;
    uint64_t demand_bytes;
    uint64_t hit_bytes;
    uint64_t worker_ready_bytes;
    double demand_ssd_read_ms_sum;
    double demand_transfer_ms_sum;
    double demand_total_ms_sum;
    double hit_transfer_ms_sum;
    double hit_total_ms_sum;
    double worker_read_ms_sum;
    double queue_to_worker_ms_sum;
    double queue_to_ready_ms_sum;
} ds4_prefetch_trace_stats;

static bool json_number(const char *line, const char *key, double *out) {
    const char *value = strstr(line, key);
    if (!value || !out) return false;
    value += strlen(key);
    char *end = NULL;
    errno = 0;
    const double parsed = strtod(value, &end);
    if (errno != 0 || end == value) return false;
    *out = parsed;
    return true;
}

static uint64_t json_u64_or_zero(const char *line, const char *key) {
    double value = 0.0;
    return json_number(line, key, &value) && value > 0.0 ? (uint64_t)value : 0;
}

static double json_double_or_zero(const char *line, const char *key) {
    double value = 0.0;
    return json_number(line, key, &value) ? value : 0.0;
}

static void read_prefetch_trace_stats(const char *path,
                                      ds4_prefetch_trace_stats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    FILE *fp = fopen(path, "r");
    char line[4096];
    if (!fp) return;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "\"event\":\"prefetch_miss_demand\"")) {
            ++stats->demand_events;
            stats->demand_bytes += json_u64_or_zero(line, "\"bytes\":");
            stats->demand_ssd_read_ms_sum += json_double_or_zero(line, "\"ssd_read_ms\":");
            stats->demand_transfer_ms_sum +=
                json_double_or_zero(line, "\"transfer_from_RAM_to_VRAM_ms\":");
            stats->demand_total_ms_sum += json_double_or_zero(line, "\"total_load_event_ms\":");
        } else if (strstr(line, "\"event\":\"prefetch_hit\"")) {
            ++stats->hit_events;
            stats->hit_bytes += json_u64_or_zero(line, "\"bytes\":");
            stats->hit_transfer_ms_sum +=
                json_double_or_zero(line, "\"transfer_from_RAM_to_VRAM_ms\":");
            stats->hit_total_ms_sum += json_double_or_zero(line, "\"total_load_event_ms\":");
        } else if (strstr(line, "\"event\":\"prefetch_enqueue\"")) {
            ++stats->enqueue_events;
        } else if (strstr(line, "\"event\":\"prefetch_worker_queued\"")) {
            ++stats->worker_queued_events;
        } else if (strstr(line, "\"event\":\"prefetch_worker_started\"")) {
            ++stats->worker_started_events;
            stats->queue_to_worker_ms_sum += json_double_or_zero(line, "\"queue_to_worker_ms\":");
        } else if (strstr(line, "\"event\":\"prefetch_worker_ready\"")) {
            ++stats->worker_ready_events;
            stats->worker_ready_bytes += json_u64_or_zero(line, "\"bytes\":");
            stats->worker_read_ms_sum += json_double_or_zero(line, "\"worker_read_ms\":");
            stats->queue_to_ready_ms_sum += json_double_or_zero(line, "\"queue_to_ready_ms\":");
        } else if (strstr(line, "\"event\":\"prefetch_worker_failed\"")) {
            ++stats->worker_failed_events;
        } else if (strstr(line, "\"event\":\"prefetch_bank_full\"")) {
            ++stats->bank_full_events;
        } else if (strstr(line, "\"event\":\"prefetch_lookup_loading\"")) {
            ++stats->lookup_loading_events;
        } else if (strstr(line, "\"event\":\"prefetch_lookup_absent\"")) {
            ++stats->lookup_absent_events;
        } else if (strstr(line, "\"event\":\"prefetch_lookup_failed\"")) {
            ++stats->lookup_failed_events;
        } else if (strstr(line, "\"event\":\"prefetch_lookup_expert_mismatch\"")) {
            ++stats->lookup_expert_mismatch_events;
        } else if (strstr(line, "\"event\":\"prefetch_stale_discarded\"")) {
            ++stats->stale_discarded_events;
        }
    }
    fclose(fp);
}

static double average_or_zero(double total, unsigned long count) {
    return count ? total / (double)count : 0.0;
}

static bool write_summary(const char *path, int a, int b, int c,
                          const char *source, unsigned layers, unsigned init_distance,
                          unsigned step,
                          const char *b_stderr, const char *c_stderr,
                          const char *b_trace, const char *c_trace,
                          const ds4_run_measurement *b_run,
                          const ds4_run_measurement *c_run) {
    time_t now = time(NULL); struct tm utc; char stamp[32];
    if (!gmtime_r(&now, &utc) || !strftime(stamp, sizeof(stamp),
                                           "%Y-%m-%dT%H:%M:%SZ", &utc)) return false;
    FILE *fp = fopen(path, "w");
    if (!fp) return false;
    ds4_prefetch_trace_stats b_stats, c_stats;
    read_prefetch_trace_stats(b_trace, &b_stats);
    read_prefetch_trace_stats(c_trace, &c_stats);
    const double b_tps = generation_tps(b_stderr);
    const double c_tps = generation_tps(c_stderr);
    const double b_demand_average = average_or_zero(
            b_stats.demand_total_ms_sum, b_stats.demand_events);
    const double c_hit_average = average_or_zero(c_stats.hit_total_ms_sum,
                                                  c_stats.hit_events);
    const double estimated_saved_ms = c_stats.hit_events *
        (b_demand_average > c_hit_average ? b_demand_average - c_hit_average : 0.0);
    const double hit_rate = c_stats.hit_events + c_stats.demand_events ?
        100.0 * (double)c_stats.hit_events /
            (double)(c_stats.hit_events + c_stats.demand_events) : 0.0;
    const double tps_delta_percent = b_tps > 0.0 ?
        100.0 * (c_tps - b_tps) / b_tps : 0.0;
    const int rc = fprintf(fp,
        "{\n  \"test\": \"%s\",\n"
        "  \"timestamp_utc\": \"%s\",\n"
        "  \"prefetch_model\": \"%s\",\n"
        "  \"prefetch_layers\": %u,\n"
        "  \"prefetch_init_distance\": %u,\n"
        "  \"prefetch_step\": %u,\n"
        "  \"run_a_exit_code\": %d,\n  \"run_b_exit_code\": %d,\n"
        "  \"run_c_exit_code\": %d,\n"
        "  \"run_b\": {\n"
        "    \"generation_tps\": %.6f, \"wall_seconds\": %.3f,\n"
        "    \"user_cpu_seconds\": %.3f, \"system_cpu_seconds\": %.3f,\n"
        "    \"demand_events\": %lu, \"demand_bytes\": %" PRIu64 ",\n"
        "    \"demand_ssd_read_ms_sum\": %.3f,\n"
        "    \"demand_transfer_from_RAM_to_VRAM_ms_sum\": %.3f,\n"
        "    \"demand_total_load_event_ms_sum\": %.3f\n"
        "  },\n"
        "  \"run_c\": {\n"
        "    \"generation_tps\": %.6f, \"wall_seconds\": %.3f,\n"
        "    \"user_cpu_seconds\": %.3f, \"system_cpu_seconds\": %.3f,\n"
        "    \"prefetch_hits\": %lu, \"prefetch_miss_demand\": %lu,\n"
        "    \"prefetch_hit_rate_percent\": %.3f,\n"
        "    \"demand_bytes\": %" PRIu64 ", \"hit_bytes\": %" PRIu64 ",\n"
        "    \"demand_ssd_read_ms_sum\": %.3f,\n"
        "    \"demand_transfer_from_RAM_to_VRAM_ms_sum\": %.3f,\n"
        "    \"demand_total_load_event_ms_sum\": %.3f,\n"
        "    \"hit_transfer_from_RAM_to_VRAM_ms_sum\": %.3f,\n"
        "    \"hit_total_load_event_ms_sum\": %.3f\n"
        "  },\n"
        "  \"prefetch_worker\": {\n"
        "    \"enqueued\": %lu, \"queued\": %lu, \"started\": %lu,\n"
        "    \"ready\": %lu, \"failed\": %lu, \"bank_full\": %lu,\n"
        "    \"lookup_loading\": %lu, \"lookup_absent\": %lu,\n"
        "    \"lookup_failed\": %lu, \"lookup_expert_mismatch\": %lu,\n"
        "    \"stale_discarded\": %lu, \"ready_bytes\": %" PRIu64 ",\n"
        "    \"worker_read_ms_sum\": %.3f, \"worker_read_ms_average\": %.3f,\n"
        "    \"queue_to_worker_ms_sum\": %.3f, \"queue_to_worker_ms_average\": %.3f,\n"
        "    \"queue_to_ready_ms_sum\": %.3f, \"queue_to_ready_ms_average\": %.3f\n"
        "  },\n"
        "  \"comparison\": {\n"
        "    \"generation_tps_delta_percent\": %.3f,\n"
        "    \"wall_seconds_delta_run_c_minus_run_b\": %.3f,\n"
        "    \"user_cpu_seconds_delta_run_c_minus_run_b\": %.3f,\n"
        "    \"system_cpu_seconds_delta_run_c_minus_run_b\": %.3f,\n"
        "    \"run_b_demand_total_load_event_ms_average\": %.3f,\n"
        "    \"run_c_hit_total_load_event_ms_average\": %.3f,\n"
        "    \"estimated_load_time_saved_by_hits_ms\": %.3f\n"
        "  }\n}\n",
        DS4_TEST_RUNNER_NAME, stamp, source, layers, init_distance, step, a, b, c,
        b_tps, b_run ? b_run->wall_seconds : 0.0,
        b_run ? b_run->user_cpu_seconds : 0.0,
        b_run ? b_run->system_cpu_seconds : 0.0,
        b_stats.demand_events, b_stats.demand_bytes,
        b_stats.demand_ssd_read_ms_sum, b_stats.demand_transfer_ms_sum,
        b_stats.demand_total_ms_sum,
        c_tps, c_run ? c_run->wall_seconds : 0.0,
        c_run ? c_run->user_cpu_seconds : 0.0,
        c_run ? c_run->system_cpu_seconds : 0.0,
        c_stats.hit_events, c_stats.demand_events, hit_rate,
        c_stats.demand_bytes, c_stats.hit_bytes,
        c_stats.demand_ssd_read_ms_sum, c_stats.demand_transfer_ms_sum,
        c_stats.demand_total_ms_sum, c_stats.hit_transfer_ms_sum,
        c_stats.hit_total_ms_sum,
        c_stats.enqueue_events, c_stats.worker_queued_events,
        c_stats.worker_started_events, c_stats.worker_ready_events,
        c_stats.worker_failed_events, c_stats.bank_full_events,
        c_stats.lookup_loading_events, c_stats.lookup_absent_events,
        c_stats.lookup_failed_events, c_stats.lookup_expert_mismatch_events,
        c_stats.stale_discarded_events, c_stats.worker_ready_bytes,
        c_stats.worker_read_ms_sum,
        average_or_zero(c_stats.worker_read_ms_sum, c_stats.worker_ready_events),
        c_stats.queue_to_worker_ms_sum,
        average_or_zero(c_stats.queue_to_worker_ms_sum, c_stats.worker_started_events),
        c_stats.queue_to_ready_ms_sum,
        average_or_zero(c_stats.queue_to_ready_ms_sum, c_stats.worker_ready_events),
        tps_delta_percent,
        (c_run ? c_run->wall_seconds : 0.0) - (b_run ? b_run->wall_seconds : 0.0),
        (c_run ? c_run->user_cpu_seconds : 0.0) - (b_run ? b_run->user_cpu_seconds : 0.0),
        (c_run ? c_run->system_cpu_seconds : 0.0) - (b_run ? b_run->system_cpu_seconds : 0.0),
        b_demand_average, c_hit_average, estimated_saved_ms);
    return rc >= 0 && fclose(fp) == 0;
}

int main(void) {
    const char *model = getenv("DS4_MODEL");
    const char *prefetch_model = getenv("DS4_TEST_5_PREFETCH_MODEL");
    struct stat model_st, prefetch_st;
    if (access("./ds4", X_OK) || !model || !model[0] || access(model, R_OK) ||
        !prefetch_model || !prefetch_model[0] || access(prefetch_model, R_OK) ||
        stat(model, &model_st) || stat(prefetch_model, &prefetch_st) ||
        model_st.st_size <= 0 || model_st.st_size != prefetch_st.st_size) {
        fprintf(stderr, "%s: set DS4_MODEL and DS4_TEST_5_PREFETCH_MODEL to equal-size GGUF files\n",
                DS4_TEST_RUNNER_LABEL);
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

    const unsigned layers = prefetch_layer_count();
    const unsigned init_distance = prefetch_init_distance();
    const unsigned step = prefetch_step();
    printf("%s: writing results to %s\n", DS4_TEST_RUNNER_LABEL, dir);
    printf("%s: demand source: %s\n", DS4_TEST_RUNNER_LABEL, model);
    printf("%s: prefetch source: %s\n", DS4_TEST_RUNNER_LABEL, prefetch_model);
    printf("%s: prefetch init distance: token t -> token t+%u; step: %u\n",
           DS4_TEST_RUNNER_LABEL, init_distance, step);
    ds4_run_measurement run_a = {0}, run_b = {0}, run_c = {0};
    printf("%s: starting Run A to create the complete routing oracle\n", DS4_TEST_RUNNER_LABEL);
    const int a = run_ds4(model, aout, aerr, oracle, NULL, aroute, atrace, NULL,
                          layers, init_distance, step, &run_a);
    if (a) { fprintf(stderr, "%s: Run A failed\n", DS4_TEST_RUNNER_LABEL); return a; }
    printf("%s: starting Run B with complete oracle replay and no prefetch\n", DS4_TEST_RUNNER_LABEL);
    const int b = run_ds4(model, bout, berr, NULL, oracle, broute, btrace, NULL,
                          layers, init_distance, step, &run_b);
    if (b) { fprintf(stderr, "%s: Run B failed\n", DS4_TEST_RUNNER_LABEL); return b; }
    printf("%s: starting Run C with complete replay and external-device prefetch "
           "(layers 0..%u, init distance %u, step %u)\n", DS4_TEST_RUNNER_LABEL,
           layers ? layers - 1u : 0u, init_distance, step);
    const int c = run_ds4(model, cout, cerr, NULL, oracle, croute, ctrace,
                          prefetch_model, layers, init_distance, step, &run_c);
    if (c) { fprintf(stderr, "%s: Run C failed\n", DS4_TEST_RUNNER_LABEL); return c; }
    if (!write_summary(summary, a, b, c, prefetch_model, layers, init_distance, step,
                       berr, cerr, btrace, ctrace, &run_b, &run_c)) return 2;
    printf("%s: Run A/B/C traces and summary written to %s\n", DS4_TEST_RUNNER_LABEL, dir);
    return 0;
}
