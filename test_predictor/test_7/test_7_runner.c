/* Small native entry point for CLion.
 *
 * CLion passes a Native Application configuration's environment only to the
 * executable it launches after Build, not to a phony Make recipe.  This
 * wrapper preserves that environment and starts the Python A/B/S/C runner.
 */

#include <stdio.h>
#include <unistd.h>

int main(void) {
    execlp("uv", "uv",
           "run", "--project", "test_predictor/prefetch_predictor_pipeline/ML_pipeline",
           "python", "test_predictor/test_7/test_7_runner.py",
           (char *)NULL);
    perror("TEST_7: impossibile avviare uv");
    return 127;
}
