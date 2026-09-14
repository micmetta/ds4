/* Entry point nativo per CLion: conserva le variabili d'ambiente configurate
 * nella Native Application e avvia il benchmark Python TEST_8. */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_8/test_8_runner.py",
           (char *)NULL);
    perror("TEST_8: impossibile avviare uv");
    return 127;
}
