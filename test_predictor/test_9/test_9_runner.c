/* Entry point nativo per CLion. TEST_9 esegue A/B/C ripetuti e, dopo ogni C,
 * salva automaticamente l'audit della banca pinned nella stessa ripetizione. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_9/test_9_runner.py",
           (char *)NULL);
    perror("TEST_9: impossibile avviare uv");
    return 127;
}
