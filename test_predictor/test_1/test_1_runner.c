/* TEST_1 starts the TEST_0 runner in oracle-routing replay mode. */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
    if (setenv("DS4_TEST_1", "1", 1) != 0) {
        perror("TEST_1: setenv");
        return 2;
    }
    execl("./test_predictor/test_0/test_0_runner",
          "./test_predictor/test_0/test_0_runner", (char *)NULL);
    perror("TEST_1: exec TEST_0 runner");
    return 2;
}
