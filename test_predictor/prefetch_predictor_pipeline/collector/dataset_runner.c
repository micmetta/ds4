/*
 * Generic dataset collector.
 *
 * Reads the V0 JSONL corpus and launches one deterministic DS4 process per
 * prompt. The parser intentionally supports the flat JSON objects used by V0:
 * {"id": number, "category": string, "prompt": string}.
 */
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    LINE_CAP = 16384,
    CATEGORY_CAP = 128,
    PROMPT_CAP = 12000,
    PATH_CAP = 4096,
};

static const char *default_input =
    "test_predictor/prefetch_predictor_pipeline/dataset_definition/prefetch_prompts_v0.jsonl";
static const char *default_corpus_id = "prefetch_prompts_v0";

static int mkdir_p(const char *path) {
    char copy[PATH_CAP];
    const size_t n = strlen(path);
    if (n == 0 || n >= sizeof(copy)) return -1;
    memcpy(copy, path, n + 1);
    for (char *p = copy + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) return -1;
        *p = '/';
    }
    return mkdir(copy, 0755) == 0 || errno == EEXIST ? 0 : -1;
}

static bool make_prompt_path(char *out, size_t cap, const char *directory,
                             unsigned long prompt_id, const char *suffix) {
    const int written = snprintf(out, cap, "%s/prompt_%04lu%s",
                                 directory, prompt_id, suffix);
    return written >= 0 && (size_t)written < cap;
}

static const char *find_json_key(const char *line, const char *key) {
    char needle[96];
    const int n = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (n < 0 || (size_t)n >= sizeof(needle)) return NULL;
    const char *p = strstr(line, needle);
    if (!p) return NULL;
    p += n;
    while (isspace((unsigned char)*p)) p++;
    return *p == ':' ? p + 1 : NULL;
}

static bool json_number(const char *line, const char *key, unsigned long *out) {
    const char *p = find_json_key(line, key);
    char *end = NULL;
    if (!p) return false;
    while (isspace((unsigned char)*p)) p++;
    const unsigned long value = strtoul(p, &end, 10);
    if (end == p) return false;
    *out = value;
    return true;
}

static bool json_string(const char *line, const char *key, char *out, size_t cap) {
    const char *p = find_json_key(line, key);
    size_t used = 0;
    if (!p) return false;
    while (isspace((unsigned char)*p)) p++;
    if (*p++ != '"') return false;

    while (*p && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c == '\\') {
            c = (unsigned char)*p++;
            if (c == 'n') c = '\n';
            else if (c == 'r') c = '\r';
            else if (c == 't') c = '\t';
            else if (c != '"' && c != '\\' && c != '/') return false;
        }
        if (used + 1 >= cap) return false;
        out[used++] = (char)c;
    }
    if (*p != '"') return false;
    out[used] = '\0';
    return true;
}

static unsigned env_unsigned(const char *name, unsigned fallback) {
    const char *text = getenv(name);
    char *end = NULL;
    if (!text || !text[0]) return fallback;
    const unsigned long value = strtoul(text, &end, 10);
    return end && *end == '\0' && value > 0 && value <= 1000000U ?
        (unsigned)value : fallback;
}

static bool env_enabled(const char *name) {
    const char *text = getenv(name);
    return text && text[0] && strcmp(text, "0") != 0;
}

static bool manifest_has_success(const char *manifest_path, unsigned long prompt_id) {
    FILE *fp = fopen(manifest_path, "r");
    char line[2048], id_text[64];
    if (!fp) return false;
    snprintf(id_text, sizeof(id_text), "\"prompt_id\":%lu", prompt_id);
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, id_text) && strstr(line, "\"status\":\"success\"")) {
            fclose(fp);
            return true;
        }
    }
    fclose(fp);
    return false;
}

static void json_write_string(FILE *fp, const char *text) {
    fputc('"', fp);
    for (; *text; text++) {
        const unsigned char c = (unsigned char)*text;
        if (c == '"' || c == '\\') fputc('\\', fp);
        if (c == '\n') fputs("\\n", fp);
        else if (c == '\r') fputs("\\r", fp);
        else if (c == '\t') fputs("\\t", fp);
        else if (c >= 0x20) fputc(c, fp);
    }
    fputc('"', fp);
}

static void append_manifest(const char *path, unsigned long prompt_id,
                            const char *category, const char *sequence_id,
                            const char *trace_path, const char *hidden_path,
                            const char *kv_path,
                            int status) {
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    fprintf(fp, "{\"prompt_id\":%lu,\"prompt_category\":", prompt_id);
    json_write_string(fp, category);
    fputs(",\"sequence_id\":", fp);
    json_write_string(fp, sequence_id);
    fputs(",\"trace_path\":", fp);
    json_write_string(fp, trace_path);
    fputs(",\"hidden_path\":", fp);
    if (hidden_path) json_write_string(fp, hidden_path);
    else fputs("null", fp);
    fputs(",\"kv_path\":", fp);
    if (kv_path) json_write_string(fp, kv_path);
    else fputs("null", fp);
    fprintf(fp, ",\"status\":\"%s\",\"exit_code\":%d}\n",
            status == 0 ? "success" : "failed", status);
    fclose(fp);
}

static int run_ds4(const char *model, const char *prompt, const char *trace_path,
                   const char *stdout_path, const char *stderr_path,
                   const char *hidden_path, const char *kv_path, unsigned max_tokens,
                   unsigned max_scores) {
    const pid_t pid = fork();
    if (pid < 0) return 127;

    if (pid == 0) {
        char token_text[16], score_text[16];
        snprintf(token_text, sizeof(token_text), "%u", max_tokens);
        snprintf(score_text, sizeof(score_text), "%u", max_scores);

        const int out = open(stdout_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        const int err = open(stderr_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 ||
            dup2(err, STDERR_FILENO) < 0 ||
            setenv("DS4_CUDA_MMQ", "0", 1) ||
            setenv("DS4_ROUTING_TRACE_PATH", trace_path, 1) ||
            setenv("DS4_DATASET_ROUTER_MAX_SCORES", score_text, 1) ||
            (hidden_path && setenv("DS4_HIDDEN_TRACE_PATH", hidden_path, 1)) ||
            (!hidden_path && unsetenv("DS4_HIDDEN_TRACE_PATH")) ||
            (kv_path && setenv("DS4_KV_TRACE_PATH", kv_path, 1)) ||
            (!kv_path && unsetenv("DS4_KV_TRACE_PATH"))) {
            _exit(127);
        }
        close(out);
        close(err);
        execl("./ds4", "./ds4", "--cuda", "--ssd-streaming",
              "--ssd-streaming-cache-experts", "8GB", "-c", "1024",
              "-n", token_text, "--nothink", "--temp", "0", "-p", prompt,
              "-m", model, (char *)NULL);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return 127;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

int main(void) {
    const char *model = getenv("DS4_MODEL");
    const char *input = getenv("DS4_DATASET_PROMPTS_PATH");
    const char *output = getenv("DS4_DATASET_OUTPUT_DIR");
    const char *corpus_id = getenv("DS4_DATASET_CORPUS_ID");
    const unsigned max_tokens = env_unsigned("DS4_DATASET_MAX_TOKENS", 64);
    const unsigned max_scores = env_unsigned("DS4_DATASET_ROUTER_MAX_SCORES", 16);
    const unsigned limit = env_unsigned("DS4_DATASET_LIMIT", 0);
    const unsigned start_id = env_unsigned("DS4_DATASET_START_ID", 1);
    const unsigned force_prompt_id = env_unsigned("DS4_DATASET_FORCE_PROMPT_ID", 0);
    const bool capture_hidden = env_enabled("DS4_DATASET_CAPTURE_HIDDEN");
    const bool capture_kv = env_enabled("DS4_DATASET_CAPTURE_KV");
    const char *build_id = getenv("DS4_DATASET_BUILD_ID");

    if (!model || !model[0]) {
        fprintf(stderr, "DATASET: DS4_MODEL must point to the GGUF model\n");
        return 2;
    }
    if (!input || !input[0]) input = default_input;
    if (!corpus_id || !corpus_id[0]) corpus_id = default_corpus_id;
    if (!output || !output[0]) {
        fprintf(stderr, "DATASET: DS4_DATASET_OUTPUT_DIR is required\n");
        return 2;
    }
    if (!build_id || !build_id[0]) build_id = "dataset";
    if (max_scores < 6 || max_scores > 256) {
        fprintf(stderr, "DATASET: DS4_DATASET_ROUTER_MAX_SCORES must be in 6..256\n");
        return 2;
    }

    char routing_dir[PATH_CAP], hidden_dir[PATH_CAP], kv_dir[PATH_CAP], logs_dir[PATH_CAP], manifest_path[PATH_CAP];
    snprintf(routing_dir, sizeof(routing_dir), "%s/routing", output);
    snprintf(hidden_dir, sizeof(hidden_dir), "%s/hidden", output);
    snprintf(kv_dir, sizeof(kv_dir), "%s/kv", output);
    snprintf(logs_dir, sizeof(logs_dir), "%s/logs", output);
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.jsonl", output);
    if (mkdir_p(routing_dir) || mkdir_p(logs_dir) ||
        (capture_hidden && mkdir_p(hidden_dir)) ||
        (capture_kv && mkdir_p(kv_dir))) {
        fprintf(stderr, "DATASET: cannot create output directories under %s: %s\n",
                output, strerror(errno));
        return 2;
    }

    FILE *corpus = fopen(input, "r");
    if (!corpus) {
        fprintf(stderr, "DATASET: cannot open %s: %s\n", input, strerror(errno));
        return 2;
    }

    char line[LINE_CAP], category[CATEGORY_CAP], prompt[PROMPT_CAP];
    unsigned processed = 0, succeeded = 0, failed = 0;
    while (fgets(line, sizeof(line), corpus)) {
        unsigned long prompt_id = 0;
        if (!json_number(line, "id", &prompt_id) ||
            !json_string(line, "category", category, sizeof(category)) ||
            !json_string(line, "prompt", prompt, sizeof(prompt))) {
            fprintf(stderr, "DATASET: skipped invalid JSONL record: %s", line);
            failed++;
            continue;
        }
        if (prompt_id < start_id) continue;
        if (force_prompt_id && prompt_id != force_prompt_id) continue;
        if (!force_prompt_id && manifest_has_success(manifest_path, prompt_id)) continue;
        if (limit && processed >= limit) break;

        char sequence_id[128], trace_path[PATH_CAP], hidden_path[PATH_CAP], kv_path[PATH_CAP];
        char stdout_path[PATH_CAP], stderr_path[PATH_CAP];
        snprintf(sequence_id, sizeof(sequence_id), "%s_%s_%lu_run_001",
                 corpus_id, build_id, prompt_id);
        if (!make_prompt_path(trace_path, sizeof(trace_path), routing_dir, prompt_id,
                              ".routing.jsonl") ||
            (capture_hidden && !make_prompt_path(hidden_path, sizeof(hidden_path),
                                                  hidden_dir, prompt_id,
                                                  ".ffn_norm.f32.bin")) ||
            (capture_kv && !make_prompt_path(kv_path, sizeof(kv_path), kv_dir, prompt_id,
                                              ".kv.f32.bin")) ||
            !make_prompt_path(stdout_path, sizeof(stdout_path), logs_dir, prompt_id,
                              ".stdout.log") ||
            !make_prompt_path(stderr_path, sizeof(stderr_path), logs_dir, prompt_id,
                              ".stderr.log")) {
            fprintf(stderr, "DATASET: output path is too long\n");
            fclose(corpus);
            return 2;
        }

        const char *prompt_hidden_path = capture_hidden ? hidden_path : NULL;
        const char *prompt_kv_path = capture_kv ? kv_path : NULL;
        fprintf(stderr, "DATASET: prompt %lu (%s)\n", prompt_id, category);
        const int status = run_ds4(model, prompt, trace_path, stdout_path, stderr_path,
                                   prompt_hidden_path, prompt_kv_path, max_tokens, max_scores);
        append_manifest(manifest_path, prompt_id, category, sequence_id, trace_path,
                        prompt_hidden_path, prompt_kv_path, status);
        processed++;
        if (status == 0) succeeded++;
        else {
            failed++;
            fprintf(stderr, "DATASET: prompt %lu failed with exit code %d\n", prompt_id, status);
        }
    }

    fclose(corpus);
    fprintf(stderr, "DATASET: processed=%u success=%u failed=%u\n",
            processed, succeeded, failed);
    return failed ? 1 : 0;
}
