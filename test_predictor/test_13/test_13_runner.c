/* Entry point nativo per CLion. TEST_13 confronta A/B/C mentre la fase C
 * legge gli expert dal servizio Windows e li riceve negli slot CUDA pinned. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_13/test_13_runner.py",
           (char *)NULL);
    perror("TEST_13: impossibile avviare uv");
    return 127;
}
