/* TEST_0: run DS4 twice in separate processes and compare decode routing. */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum { TOP_K = 6, TOP_CANDIDATES = 8, ORACLE_MAX_EXPERT_USED = 8, LINE_CAP = 2048 };

typedef struct {
    int expert;
    float prob;
    float bias;
    float selection_score;
} routing_candidate;

typedef struct {
    uint32_t token_index;
    uint32_t token_id;
    uint32_t layer;
    int experts[TOP_K];
    bool has_scores;
    routing_candidate candidates[TOP_CANDIDATES];
    float cutoff_margin;
} routing_event;

typedef struct {
    routing_event *items;
    size_t count;
    size_t cap;
} event_vec;

typedef struct {
    uint32_t token_index;
    uint32_t token_id;
} token_event;

typedef struct {
    token_event *items;
    size_t count;
    size_t cap;
} token_vec;

typedef struct {
    uint32_t token_index;
    uint32_t layer;
    bool has_a;
    bool has_b;
    int experts_a[TOP_K];
    int experts_b[TOP_K];
    bool has_scores_a;
    bool has_scores_b;
    routing_candidate candidates_a[TOP_CANDIDATES];
    routing_candidate candidates_b[TOP_CANDIDATES];
    float cutoff_margin_a;
    float cutoff_margin_b;
} routing_mismatch;

typedef struct {
    routing_mismatch *items;
    size_t count;
    size_t cap;
} mismatch_vec;

typedef struct {
    uint32_t token_index;
    bool has_a;
    bool has_b;
    uint32_t token_id_a;
    uint32_t token_id_b;
} token_mismatch;

typedef struct {
    token_mismatch *items;
    size_t count;
    size_t cap;
} token_mismatch_vec;

static const char *output_root = "test_predictor/test_0/output";
static char output_dir[PATH_MAX];
static char trace_a[PATH_MAX];
static char trace_b[PATH_MAX];
static char stdout_a[PATH_MAX];
static char stdout_b[PATH_MAX];
static char stderr_a[PATH_MAX];
static char stderr_b[PATH_MAX];
static char comparison_path[PATH_MAX];
static char oracle_path[PATH_MAX];
static char oracle_json_path[PATH_MAX];

typedef struct {
    uint32_t token_index;
    uint32_t layer;
    int32_t experts[ORACLE_MAX_EXPERT_USED];
    float weights[ORACLE_MAX_EXPERT_USED];
} oracle_binary_entry;

static const unsigned char oracle_magic[8] = { 'D', 'S', '4', 'O', 'R', 'C', '1', '\0' };

static bool grow(void **items, size_t *cap, size_t item_size) {
    const size_t next = *cap ? *cap * 2u : 128u;
    if (next < *cap || next > SIZE_MAX / item_size) return false;
    void *p = realloc(*items, next * item_size);
    if (!p) return false;
    *items = p;
    *cap = next;
    return true;
}

static bool push_event(event_vec *v, const routing_event *event) {
    if (v->count == v->cap && !grow((void **)&v->items, &v->cap, sizeof(*v->items))) return false;
    v->items[v->count++] = *event;
    return true;
}

static bool push_token(token_vec *v, token_event event) {
    if (v->count == v->cap && !grow((void **)&v->items, &v->cap, sizeof(*v->items))) return false;
    v->items[v->count++] = event;
    return true;
}

static bool push_mismatch(mismatch_vec *v, const routing_event *a, const routing_event *b) {
    if (v->count == v->cap && !grow((void **)&v->items, &v->cap, sizeof(*v->items))) return false;
    routing_mismatch *m = &v->items[v->count++];
    memset(m, 0, sizeof(*m));
    m->has_a = a != NULL;
    m->has_b = b != NULL;
    m->token_index = a ? a->token_index : b->token_index;
    m->layer = a ? a->layer : b->layer;
    if (a) memcpy(m->experts_a, a->experts, sizeof(m->experts_a));
    if (b) memcpy(m->experts_b, b->experts, sizeof(m->experts_b));
    if (a) {
        m->has_scores_a = a->has_scores;
        memcpy(m->candidates_a, a->candidates, sizeof(m->candidates_a));
        m->cutoff_margin_a = a->cutoff_margin;
    }
    if (b) {
        m->has_scores_b = b->has_scores;
        memcpy(m->candidates_b, b->candidates, sizeof(m->candidates_b));
        m->cutoff_margin_b = b->cutoff_margin;
    }
    return true;
}

static bool push_token_mismatch(token_mismatch_vec *v, const token_event *a, const token_event *b) {
    if (v->count == v->cap && !grow((void **)&v->items, &v->cap, sizeof(*v->items))) return false;
    token_mismatch *m = &v->items[v->count++];
    m->has_a = a != NULL;
    m->has_b = b != NULL;
    m->token_index = a ? a->token_index : b->token_index;
    m->token_id_a = a ? a->token_id : 0;
    m->token_id_b = b ? b->token_id : 0;
    return true;
}

static int event_cmp(const void *left, const void *right) {
    const routing_event *a = left;
    const routing_event *b = right;
    if (a->token_index != b->token_index) return a->token_index < b->token_index ? -1 : 1;
    if (a->layer != b->layer) return a->layer < b->layer ? -1 : 1;
    return 0;
}

static bool same_key(const routing_event *a, const routing_event *b) {
    return a->token_index == b->token_index && a->layer == b->layer;
}

static bool parse_event(const char *line, routing_event *event) {
    char phase[16] = {0};
    char is_hash[6] = {0};
    int offset = 0;
    const int found = sscanf(line,
        "{\"phase\":\"%15[^\"]\",\"token_index\":%" SCNu32
        ",\"token_id\":%" SCNu32 ",\"layer\":%" SCNu32
        ",\"is_hash\":%5[^,],\"experts\":[%d,%d,%d,%d,%d,%d]%n",
        phase, &event->token_index, &event->token_id, &event->layer, is_hash,
        &event->experts[0], &event->experts[1], &event->experts[2],
        &event->experts[3], &event->experts[4], &event->experts[5], &offset);
    if (found != 11 || strcmp(phase, "decode") != 0 ||
        (strcmp(is_hash, "true") != 0 && strcmp(is_hash, "false") != 0)) return false;

    const char *cursor = line + offset;
    if (*cursor == '}') {
        event->has_scores = false;
        return true;
    }
    if (*cursor++ != ',') return false;
    const char *const no_scores = "\"top_candidates\":null,\"cutoff_margin\":null";
    if (strncmp(cursor, no_scores, strlen(no_scores)) == 0) {
        event->has_scores = false;
        return true;
    }
    if (strncmp(cursor, "\"top_candidates\":[", 18) != 0) return false;
    cursor += 18;
    event->has_scores = true;
    for (int i = 0; i < TOP_CANDIDATES; i++) {
        int used = 0;
        const int parsed = sscanf(cursor, "[%d,%f,%f,%f]%n",
                                  &event->candidates[i].expert,
                                  &event->candidates[i].prob,
                                  &event->candidates[i].bias,
                                  &event->candidates[i].selection_score,
                                  &used);
        if (parsed != 4 || used <= 0) return false;
        cursor += used;
        if (i + 1 < TOP_CANDIDATES) {
            if (*cursor++ != ',') return false;
        }
    }
    int used = 0;
    if (sscanf(cursor, "],\"cutoff_margin\":%f%n", &event->cutoff_margin, &used) != 1 ||
        used <= 0 || (cursor[used] != '\0' && cursor[used] != '\n' && cursor[used] != ',')) return false;
    return true;
}

static bool load_trace(const char *path, event_vec *events) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "TEST_0: cannot read %s: %s\n", path, strerror(errno));
        return false;
    }
    char line[LINE_CAP];
    unsigned line_no = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_no++;
        routing_event event;
        if (!parse_event(line, &event)) {
            fprintf(stderr, "TEST_0: invalid JSONL event at %s:%u\n", path, line_no);
            fclose(fp);
            return false;
        }
        if (!push_event(events, &event)) {
            fprintf(stderr, "TEST_0: out of memory while reading %s\n", path);
            fclose(fp);
            return false;
        }
    }
    if (ferror(fp) || fclose(fp) != 0 || events->count == 0) {
        fprintf(stderr, "TEST_0: empty or unreadable trace %s\n", path);
        return false;
    }
    qsort(events->items, events->count, sizeof(events->items[0]), event_cmp);
    for (size_t i = 1; i < events->count; i++) {
        if (same_key(&events->items[i - 1], &events->items[i])) {
            fprintf(stderr, "TEST_0: duplicate routing event in %s\n", path);
            return false;
        }
    }
    return true;
}

static bool collect_tokens(const event_vec *events, token_vec *tokens, const char *name) {
    for (size_t i = 0; i < events->count; i++) {
        const routing_event *event = &events->items[i];
        if (tokens->count && tokens->items[tokens->count - 1].token_index == event->token_index) {
            if (tokens->items[tokens->count - 1].token_id != event->token_id) {
                fprintf(stderr, "TEST_0: inconsistent token id in %s at token_index=%" PRIu32 "\n",
                        name, event->token_index);
                return false;
            }
            continue;
        }
        if (!push_token(tokens, (token_event) { event->token_index, event->token_id })) return false;
    }
    return true;
}

static bool experts_equal_ordered(const routing_event *a, const routing_event *b) {
    return memcmp(a->experts, b->experts, sizeof(a->experts)) == 0;
}

static int int_cmp(const void *left, const void *right) {
    const int a = *(const int *)left;
    const int b = *(const int *)right;
    return (a > b) - (a < b);
}

static bool experts_equal_set(const routing_event *a, const routing_event *b) {
    int sorted_a[TOP_K];
    int sorted_b[TOP_K];
    memcpy(sorted_a, a->experts, sizeof(sorted_a));
    memcpy(sorted_b, b->experts, sizeof(sorted_b));
    qsort(sorted_a, TOP_K, sizeof(sorted_a[0]), int_cmp);
    qsort(sorted_b, TOP_K, sizeof(sorted_b[0]), int_cmp);
    return memcmp(sorted_a, sorted_b, sizeof(sorted_a)) == 0;
}

static bool files_equal(const char *left, const char *right, bool *equal) {
    FILE *a = fopen(left, "rb");
    FILE *b = fopen(right, "rb");
    if (!a || !b) {
        fprintf(stderr, "TEST_0: cannot compare stdout files: %s\n", strerror(errno));
        if (a) fclose(a);
        if (b) fclose(b);
        return false;
    }
    int ca, cb;
    do {
        ca = fgetc(a);
        cb = fgetc(b);
    } while (ca == cb && ca != EOF);
    *equal = ca == cb;
    const bool ok = !ferror(a) && !ferror(b) && fclose(a) == 0 && fclose(b) == 0;
    return ok;
}

static void write_experts(FILE *fp, const int experts[TOP_K]) {
    fprintf(fp, "[%d,%d,%d,%d,%d,%d]", experts[0], experts[1], experts[2],
            experts[3], experts[4], experts[5]);
}

static void write_score_summary(FILE *fp,
                                bool has_scores,
                                const routing_candidate candidates[TOP_CANDIDATES],
                                float cutoff_margin) {
    if (!has_scores) {
        fputs("null", fp);
        return;
    }
    fputs("{\"top_candidates\":[", fp);
    for (int i = 0; i < TOP_CANDIDATES; i++) {
        if (i) fputc(',', fp);
        fprintf(fp, "{\"expert\":%d,\"prob\":%.9g,\"bias\":%.9g,\"selection_score\":%.9g}",
                candidates[i].expert, candidates[i].prob, candidates[i].bias,
                candidates[i].selection_score);
    }
    fprintf(fp, "],\"cutoff_margin\":%.9g}", cutoff_margin);
}

static void print_experts(const int experts[TOP_K]) {
    write_experts(stdout, experts);
}

static void write_mismatches(FILE *fp, const mismatch_vec *mismatches) {
    fputs("[", fp);
    for (size_t i = 0; i < mismatches->count; i++) {
        const routing_mismatch *m = &mismatches->items[i];
        if (i) fputs(",\n", fp);
        fprintf(fp, "{\"token_index\":%" PRIu32 ",\"layer\":%" PRIu32
                ",\"experts_run_A\":", m->token_index, m->layer);
        if (m->has_a) write_experts(fp, m->experts_a); else fputs("null", fp);
        fputs(",\"experts_run_B\":", fp);
        if (m->has_b) write_experts(fp, m->experts_b); else fputs("null", fp);
        fputs(",\"router_scores_run_A\":", fp);
        write_score_summary(fp, m->has_scores_a, m->candidates_a, m->cutoff_margin_a);
        fputs(",\"router_scores_run_B\":", fp);
        write_score_summary(fp, m->has_scores_b, m->candidates_b, m->cutoff_margin_b);
        fputs("}", fp);
    }
    fputs("]", fp);
}

static void write_token_mismatches(FILE *fp, const token_mismatch_vec *mismatches) {
    fputs("[", fp);
    for (size_t i = 0; i < mismatches->count; i++) {
        const token_mismatch *m = &mismatches->items[i];
        if (i) fputs(",\n", fp);
        fprintf(fp, "{\"token_index\":%" PRIu32 ",\"token_id_run_A\":", m->token_index);
        if (m->has_a) fprintf(fp, "%" PRIu32, m->token_id_a); else fputs("null", fp);
        fputs(",\"token_id_run_B\":", fp);
        if (m->has_b) fprintf(fp, "%" PRIu32, m->token_id_b); else fputs("null", fp);
        fputs("}", fp);
    }
    fputs("]", fp);
}

static bool write_comparison(bool test_1,
                             bool pass,
                             const char *executed_at_utc,
                             bool no_tf32,
                             bool quality_mode,
                             size_t events_a,
                             size_t events_b,
                             size_t total_events,
                             bool token_match,
                             size_t token_matches,
                             size_t total_tokens,
                             bool stdout_match,
                             size_t ordered_matches,
                             size_t set_matches,
                             const token_mismatch_vec *token_mismatches,
                             const mismatch_vec *order_mismatches,
                             const mismatch_vec *set_mismatches) {
    FILE *fp = fopen(comparison_path, "w");
    if (!fp) return false;
    const double token_pct = total_tokens ? 100.0 * (double)token_matches / total_tokens : 0.0;
    const double order_pct = total_events ? 100.0 * (double)ordered_matches / total_events : 0.0;
    const double set_pct = total_events ? 100.0 * (double)set_matches / total_events : 0.0;
    fprintf(fp,
            "{\n"
            "  \"test\": \"%s\",\n"
            "  \"executed_at_utc\": \"%s\",\n"
            "  \"cuda_tf32_disabled\": %s,\n"
            "  \"quality_mode\": %s,\n"
            "  \"pass\": %s,\n"
            "  \"pass_rule\": \"token sequence, event count, and expert-set routing must match exactly\",\n"
            "  \"run_A_events\": %zu,\n"
            "  \"run_B_events\": %zu,\n"
            "  \"total_routing_events\": %zu,\n"
            "  \"token_sequence_match\": %s,\n"
            "  \"token_sequence_match_percent\": %.6f,\n"
            "  \"stdout_exact_match\": %s,\n"
            "  \"ordered_routing_match_percent\": %.6f,\n"
            "  \"expert_set_routing_match_percent\": %.6f,\n"
            "  \"exact_order_matches\": %zu,\n"
            "  \"set_matches\": %zu,\n"
            "  \"order_mismatches\": %zu,\n"
            "  \"set_mismatches\": %zu,\n"
            "  \"token_mismatches\": ",
            test_1 ? "TEST_1 forced routing replay" : "TEST_0 routing determinism",
            executed_at_utc, no_tf32 ? "true" : "false", quality_mode ? "true" : "false",
            pass ? "true" : "false",
            events_a, events_b, total_events,
            token_match ? "true" : "false", token_pct,
            stdout_match ? "true" : "false", order_pct, set_pct,
            ordered_matches, set_matches, order_mismatches->count,
            set_mismatches->count);
    write_token_mismatches(fp, token_mismatches);
    fputs(",\n  \"order_mismatch_details\": ", fp);
    write_mismatches(fp, order_mismatches);
    fputs(",\n  \"set_mismatch_details\": ", fp);
    write_mismatches(fp, set_mismatches);
    fputs("\n}\n", fp);
    return fclose(fp) == 0;
}

static bool run_ds4(const char *label, const char *trace, const char *stdout_path,
                    const char *stderr_path, bool no_tf32, bool quality_mode,
                    bool test_1) {
    const char *model = getenv("DS4_MODEL");
    const char *n_predict = getenv("DS4_TEST_0_N");
    const char *prompt = getenv("DS4_TEST_0_PROMPT");
    if (!model || !model[0]) model = "ds4flash.gguf";
    if (!n_predict || !n_predict[0]) n_predict = "64";
    if (!prompt || !prompt[0]) prompt = "Explain what a binary search is in one short paragraph.";

    const char *const test_label = test_1 ? "TEST_1" : "TEST_0";
    if (test_1) {
        printf("TEST_1: starting Run %s via shared TEST_0 runner%s\n", label,
               strcmp(label, "B") == 0 ? " with oracle replay" : "");
    } else {
        printf("TEST_0: starting Run %s as a new DS4 process\n", label);
    }
    fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        const int out = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0 ||
            setenv("DS4_CUDA_MMQ", "0", 1) != 0 || setenv("DS4_ROUTING_TRACE_PATH", trace, 1) != 0 ||
            (no_tf32 && setenv("DS4_CUDA_NO_TF32", "1", 1) != 0)) {
            dprintf(STDERR_FILENO, "%s: child setup failed: %s\n", test_label, strerror(errno));
            _exit(127);
        }
        if (test_1 && setenv(strcmp(label, "A") == 0 ?
                              "DS4_ROUTING_ORACLE_WRITE_PATH" :
                              "DS4_ROUTING_ORACLE_REPLAY_PATH",
                              oracle_path, 1) != 0) {
            dprintf(STDERR_FILENO, "TEST_1: child oracle setup failed: %s\n", strerror(errno));
            _exit(127);
        }
        if (out > STDERR_FILENO) close(out);
        if (err > STDERR_FILENO) close(err);
        char *const args[] = {
            "./ds4", "--cuda", "--ssd-streaming", "--ssd-streaming-cache-experts", "8GB",
            "-m", (char *)model, "-c", "1024", "-n", (char *)n_predict, "--nothink",
            "--temp", "0", "-p", (char *)prompt, NULL,
        };
        char *const quality_args[] = {
            "./ds4", "--cuda", "--quality", "--ssd-streaming", "--ssd-streaming-cache-experts", "8GB",
            "-m", (char *)model, "-c", "1024", "-n", (char *)n_predict, "--nothink",
            "--temp", "0", "-p", (char *)prompt, NULL,
        };
        execv(args[0], quality_mode ? quality_args : args);
        dprintf(STDERR_FILENO, "%s: exec ./ds4 failed: %s\n", test_label, strerror(errno));
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "%s: Run %s failed\n", test_label, label);
        return false;
    }
    struct stat st;
    if (stat(trace, &st) != 0 || st.st_size == 0) {
        fprintf(stderr, "%s: Run %s produced no routing trace\n", test_label, label);
        return false;
    }
    return true;
}

static bool set_output_paths(const char *dir) {
    const int written = snprintf(output_dir, sizeof(output_dir), "%s", dir);
    if (written < 0 || (size_t)written >= sizeof(output_dir)) return false;

    const char *const names[] = {
        "run_A.jsonl", "run_B.jsonl", "run_A.stdout.txt", "run_B.stdout.txt",
        "run_A.stderr.txt", "run_B.stderr.txt", "comparison.json",
    };
    char *const paths[] = {
        trace_a, trace_b, stdout_a, stdout_b, stderr_a, stderr_b, comparison_path,
    };
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        const int path_written = snprintf(paths[i], PATH_MAX, "%s/%s", output_dir, names[i]);
        if (path_written < 0 || path_written >= PATH_MAX) return false;
    }
    const int oracle_written = snprintf(oracle_path, sizeof(oracle_path), "%s/routing_oracle.bin", output_dir);
    if (oracle_written < 0 || (size_t)oracle_written >= sizeof(oracle_path)) return false;
    const int oracle_json_written = snprintf(oracle_json_path, sizeof(oracle_json_path),
                                             "%s/routing_oracle.json", output_dir);
    if (oracle_json_written < 0 || (size_t)oracle_json_written >= sizeof(oracle_json_path)) return false;
    return true;
}

static bool write_oracle_json(void) {
    FILE *bin = fopen(oracle_path, "rb");
    FILE *json = NULL;
    unsigned char magic[sizeof(oracle_magic)];
    if (!bin || fread(magic, 1, sizeof(magic), bin) != sizeof(magic) ||
        memcmp(magic, oracle_magic, sizeof(magic)) != 0) goto fail;
    json = fopen(oracle_json_path, "w");
    if (!json) goto fail;

    fputs("{\n  \"format\": \"TEST_1 routing oracle\",\n  \"entries\": [\n", json);
    oracle_binary_entry entry;
    size_t count = 0;
    while (fread(&entry, sizeof(entry), 1, bin) == 1) {
        if (count++) fputs(",\n", json);
        fprintf(json, "    {\"token_index\":%" PRIu32 ",\"layer\":%" PRIu32
                      ",\"experts\":[%d,%d,%d,%d,%d,%d],\"router_weights\":[",
                entry.token_index, entry.layer,
                entry.experts[0], entry.experts[1], entry.experts[2],
                entry.experts[3], entry.experts[4], entry.experts[5]);
        for (int i = 0; i < TOP_K; i++) {
            if (i) fputc(',', json);
            fprintf(json, "%.9g", entry.weights[i]);
        }
        fputs("]}", json);
    }
    if (ferror(bin) || fputs("\n  ]\n}\n", json) == EOF || fclose(json) != 0 || fclose(bin) != 0) {
        return false;
    }
    printf("TEST_1: routing oracle JSON written to %s (%zu entries)\n", oracle_json_path, count);
    return true;

fail:
    if (json) fclose(json);
    if (bin) fclose(bin);
    fprintf(stderr, "TEST_1: cannot convert %s to JSON: %s\n", oracle_path, strerror(errno));
    return false;
}

static bool create_next_output_dir(void) {
    char candidate[PATH_MAX];
    for (unsigned index = 1; index != 0; index++) {
        const int written = index == 1
            ? snprintf(candidate, sizeof(candidate), "%s", output_root)
            : snprintf(candidate, sizeof(candidate), "%s_%u", output_root, index);
        if (written < 0 || (size_t)written >= sizeof(candidate)) return false;
        if (mkdir(candidate, 0755) == 0) return set_output_paths(candidate);
        if (errno != EEXIST) {
            fprintf(stderr, "TEST_0: cannot create %s: %s\n", candidate, strerror(errno));
            return false;
        }
    }
    return false;
}

/* CLion can accidentally retain a trailing space in an environment value. */
static bool normalize_model_env(char *model, size_t cap) {
    const char *value = getenv("DS4_MODEL");
    if (!value || !value[0]) value = "ds4flash.gguf";

    while (isspace((unsigned char)*value)) value++;
    const size_t len = strlen(value);
    size_t end = len;
    while (end > 0 && isspace((unsigned char)value[end - 1])) end--;
    if (end == 0 || end >= cap) return false;

    memcpy(model, value, end);
    model[end] = '\0';
    return setenv("DS4_MODEL", model, 1) == 0;
}

static bool current_utc_timestamp(char *timestamp, size_t cap) {
    const time_t now = time(NULL);
    struct tm utc;
    return now != (time_t)-1 && gmtime_r(&now, &utc) != NULL &&
           strftime(timestamp, cap, "%Y-%m-%dT%H:%M:%SZ", &utc) != 0;
}

static bool env_enabled(const char *name) {
    const char *const value = getenv(name);
    return value && value[0] && strcmp(value, "0") != 0;
}

int main(int argc, char **argv) {
    const bool compare_only = argc >= 2 && strcmp(argv[1], "--compare-only") == 0;
    if (argc > 3 || (argc > 1 && !compare_only)) {
        fprintf(stderr, "usage: %s [--compare-only [output_directory]]\n", argv[0]);
        return 2;
    }
    const bool test_1 = env_enabled("DS4_TEST_1");
    const char *const test_label = test_1 ? "TEST_1" : "TEST_0";
    if (test_1) output_root = "test_predictor/test_1/output";
    if (compare_only) {
        const char *const dir = argc == 3 ? argv[2] : output_root;
        if (!set_output_paths(dir)) {
            fprintf(stderr, "TEST_0: output path is too long\n");
            return 2;
        }
    } else if (!create_next_output_dir()) {
        fprintf(stderr, "TEST_0: cannot select a new output directory\n");
        return 2;
    }
    char executed_at_utc[32];
    if (!current_utc_timestamp(executed_at_utc, sizeof(executed_at_utc))) {
        fprintf(stderr, "TEST_0: cannot create execution timestamp\n");
        return 2;
    }
    char model[LINE_CAP];
    const bool no_tf32 = env_enabled("DS4_TEST_0_NO_TF32");
    const bool quality_mode = env_enabled("DS4_TEST_0_QUALITY");
    if (!normalize_model_env(model, sizeof(model))) {
        fprintf(stderr, "TEST_0: invalid DS4_MODEL value\n");
        return 2;
    }
    if (!compare_only && access("./ds4", X_OK) != 0) {
        fprintf(stderr, "TEST_0: cannot find executable ./ds4; working directory must be /home/michele/ds4\n");
        return 2;
    }
    if (!compare_only && access(model, R_OK) != 0) {
        fprintf(stderr, "TEST_0: cannot read model file: %s\n", model);
        return 2;
    }
    if (!compare_only) {
        printf("TEST_%d: writing results to %s\n", test_1 ? 1 : 0, output_dir);
        if (no_tf32) printf("%s: TF32 disabled for both runs\n", test_label);
        if (quality_mode) printf("%s: quality mode enabled for both runs\n", test_label);
        if (!run_ds4("A", trace_a, stdout_a, stderr_a, no_tf32, quality_mode, test_1)) return 2;
        if (test_1) {
            printf("TEST_1: converting Run A routing oracle to readable JSON\n");
            if (!write_oracle_json()) return 2;
        }
        if (!run_ds4("B", trace_b, stdout_b, stderr_b, no_tf32, quality_mode, test_1)) return 2;
    }

    event_vec a = {0}, b = {0};
    token_vec tokens_a = {0}, tokens_b = {0};
    mismatch_vec order_mismatches = {0}, set_mismatches = {0};
    token_mismatch_vec token_mismatches = {0};
    bool stdout_match = false;
    const bool ok = load_trace(trace_a, &a) && load_trace(trace_b, &b) &&
                    collect_tokens(&a, &tokens_a, "Run A") && collect_tokens(&b, &tokens_b, "Run B") &&
                    files_equal(stdout_a, stdout_b, &stdout_match);
    if (!ok) return 2;

    size_t ia = 0, ib = 0, total = 0, ordered = 0, set_matches = 0;
    while (ia < a.count || ib < b.count) {
        const routing_event *left = ia < a.count ? &a.items[ia] : NULL;
        const routing_event *right = ib < b.count ? &b.items[ib] : NULL;
        const int cmp = !left ? 1 : !right ? -1 : event_cmp(left, right);
        total++;
        if (cmp == 0) {
            if (experts_equal_ordered(left, right)) ordered++;
            else if (!push_mismatch(&order_mismatches, left, right)) return 2;
            if (experts_equal_set(left, right)) set_matches++;
            else if (!push_mismatch(&set_mismatches, left, right)) return 2;
            ia++; ib++;
        } else if (cmp < 0) {
            if (!push_mismatch(&order_mismatches, left, NULL) || !push_mismatch(&set_mismatches, left, NULL)) return 2;
            ia++;
        } else {
            if (!push_mismatch(&order_mismatches, NULL, right) || !push_mismatch(&set_mismatches, NULL, right)) return 2;
            ib++;
        }
    }

    size_t ita = 0, itb = 0, token_matches = 0;
    while (ita < tokens_a.count || itb < tokens_b.count) {
        const token_event *left = ita < tokens_a.count ? &tokens_a.items[ita] : NULL;
        const token_event *right = itb < tokens_b.count ? &tokens_b.items[itb] : NULL;
        if (left && right && left->token_index == right->token_index) {
            if (left->token_id == right->token_id) token_matches++;
            else if (!push_token_mismatch(&token_mismatches, left, right)) return 2;
            ita++; itb++;
        } else if (!right || (left && left->token_index < right->token_index)) {
            if (!push_token_mismatch(&token_mismatches, left, NULL)) return 2;
            ita++;
        } else {
            if (!push_token_mismatch(&token_mismatches, NULL, right)) return 2;
            itb++;
        }
    }

    const bool token_match = token_mismatches.count == 0;
    const bool pass = token_match && a.count == b.count && set_matches == total;
    const size_t total_tokens = tokens_a.count > tokens_b.count ? tokens_a.count : tokens_b.count;
    if (!write_comparison(test_1, pass, executed_at_utc, no_tf32, quality_mode, a.count, b.count, total, token_match, token_matches, total_tokens,
                          stdout_match, ordered, set_matches, &token_mismatches,
                          &order_mismatches, &set_mismatches)) {
        fprintf(stderr, "%s: failed to write %s\n", test_label, comparison_path);
        return 2;
    }
    printf("%s: events=%zu tokens=%.3f%% ordered=%.3f%% sets=%.3f%%\n", test_label,
           total,
           total_tokens ? 100.0 * (double)token_matches / total_tokens : 0.0,
           total ? 100.0 * (double)ordered / total : 0.0,
           total ? 100.0 * (double)set_matches / total : 0.0);
    for (size_t i = 0; i < order_mismatches.count && i < 10; i++) {
        const routing_mismatch *m = &order_mismatches.items[i];
        printf("%s mismatch: token_index=%" PRIu32 " layer=%" PRIu32 " Run A=", test_label,
               m->token_index, m->layer);
        if (m->has_a) print_experts(m->experts_a); else fputs("null", stdout);
        fputs(" Run B=", stdout);
        if (m->has_b) print_experts(m->experts_b); else fputs("null", stdout);
        fputc('\n', stdout);
    }
    printf("%s: %s\n", test_label, pass ? "PASS" : "FAIL");
    free(a.items); free(b.items); free(tokens_a.items); free(tokens_b.items);
    free(order_mismatches.items); free(set_mismatches.items); free(token_mismatches.items);
    return pass ? 0 : 1;
}
