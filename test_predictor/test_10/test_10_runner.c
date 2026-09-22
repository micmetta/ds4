/* Entry point nativo per CLion. TEST_10 misura solamente il percorso I/O
 * Linux -> /mnt/d -> SSD esterno, senza inizializzare DS4, CUDA o TensorFlow. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_10/test_10_runner.py",
           (char *)NULL);
    perror("TEST_10: impossibile avviare uv");
    return 127;
}
