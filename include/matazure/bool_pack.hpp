#pragma once

#include <type_traits>

namespace matazure {

template <typename... Args>
MATAZURE_GENERAL inline bool all(Args... args);

MATAZURE_GENERAL inline bool all(bool b0) { return b0; }
MATAZURE_GENERAL inline bool all(bool b0, bool b1) { return b0 && b1; }
MATAZURE_GENERAL inline bool all(bool b0, bool b1, bool b2) { return b0 && b1 && b2; }
MATAZURE_GENERAL inline bool all(bool b0, bool b1, bool b2, bool b3) {
    return b0 && b1 && b2 && b3;
}
MATAZURE_GENERAL inline bool all(bool b0, bool b1, bool b2, bool b3, bool b4) {
    return b0 && b1 && b2 && b3 && b4;
}
MATAZURE_GENERAL inline bool all(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5) {
    return b0 && b1 && b2 && b3 && b4 && b5;
}

}  // namespace matazure