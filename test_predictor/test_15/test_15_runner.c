/* Entry point nativo per CLion. TEST_15 confronta la cache condivisa di
 * expert in RAM pinned con il replay baseline della configurazione record. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_15/test_15_runner.py",
           (char *)NULL);
    perror("TEST_15: impossibile avviare uv");
    return 127;
}
