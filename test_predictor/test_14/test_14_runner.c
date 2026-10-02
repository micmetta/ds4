/* Entry point nativo per CLion.  TEST_14 riusa A/B/C di TEST_13 con il
 * predictor Top-4 di Build 7 nella sola fase C. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_14/test_14_runner.py",
           (char *)NULL);
    perror("TEST_14: impossibile avviare uv");
    return 127;
}
