#include <matazure/config.hpp>

namespace matazure {

class invalid_shape : public std::runtime_error {
   public:
    invalid_shape() : std::runtime_error("the shape is inavlid") {}
};

}  // namespace matazure
