// main.cpp
#include "frugally-deep/include/fdeep/fdeep.hpp"
int main()
{
    const auto model = fdeep::load_model("fdeep_model.json");
    const auto result = model.predict(
        {fdeep::tensor3(fdeep::shape3(4, 1, 1), {1, 2, 3, 4})});
    std::cout << fdeep::show_tensor3s(result) << std::endl;
}
