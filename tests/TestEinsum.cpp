#include <gtest/gtest.h>

#include "einsum/Einsum.hpp"

using sparsetensor::tensor::shape_t;
using sparsetensor::einsum::raw_subscript;
using sparsetensor::einsum::einsum;


TEST(TestEinsum, simple_call) {
    shape_t shape{2, 2};

    HyperTrieTensor tensor_0{shape};
    tensor_0.set({0, 0}, 1);
    tensor_0.set({0, 1}, 2);
    tensor_0.set({1, 0}, 3);
    tensor_0.set({1, 1}, 5);

    HyperTrieTensor tensor_1{shape};
    tensor_1.set({0, 0}, 7);
    tensor_1.set({0, 1}, 11);
    tensor_1.set({1, 0}, 13);
    tensor_1.set({1, 1}, 17);

    vector<HyperTrieTensor *> operands{&tensor_0, &tensor_1};

    vector<raw_subscript> op_sc{{0, 1},
                                {1, 2}};
    raw_subscript res_sc{0, 2};

    CrossProductTensor<int> *const result = einsum<int>(operands, op_sc, res_sc);


}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}