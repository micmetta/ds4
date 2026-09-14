/* TEST_6 changes only the scheduling policy of TEST_5.
 * The shared runner preserves the exact A/B/C process orchestration. */
#define DS4_TEST_RUNNER_LABEL "TEST_6"
#define DS4_TEST_RUNNER_NAME "TEST_6 stepped external-device pinned-RAM prefetch"
#define DS4_TEST_RUNNER_DIR "test_predictor/test_6"
#define DS4_TEST_RUNNER_INIT_DISTANCE_ENV "DS4_TEST_5_PREFETCH_INIT_DISTANCE"
#define DS4_TEST_RUNNER_STEP_ENV "DS4_TEST_5_PREFETCH_STEP"
#include "../test_5/test_5_runner.c"
