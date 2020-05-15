#include "ut_foundation.hpp"
#include "ut_tensor.hpp"
#include "view/ut_permute.hpp"
#include "view/ut_slice.hpp"
#include "view/ut_zip.hpp"

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
