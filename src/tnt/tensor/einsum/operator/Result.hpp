#ifndef SPARSETENSOR_EINSUM_OPERATOR_RESULT
#define SPARSETENSOR_EINSUM_OPERATOR_RESULT

#include <algorithm>
#include <numeric>
#include <vector>
#include <memory>

#include "tnt/tensor/hypertrie/BoolHyperTrie.hpp"
#include "tnt/tensor/hypertrie/Join.hpp"
#include "tnt/tensor/einsum/operator/Einsum.hpp"
#include "tnt/tensor/einsum/operator/CrossProduct.hpp"
#include "tnt/tensor/einsum/EinsumPlan.hpp"
#include "tnt/util/container/NDMap.hpp"
#include "tnt/util/All.hpp"

namespace tnt::tensor::einsum::operators {
    namespace {
        template<typename V>
        using NDMap = std::set<Key_t, V>;
    };

    // TODO: reeanble
    template<typename T>
    class Result {
        using BoolHyperTrie = tnt::tensor::hypertrie::BoolHyperTrie;
        using Join = tnt::tensor::hypertrie::Join;
        using NewJoin = tnt::tensor::hypertrie::Join;
        using Operands = tnt::tensor::hypertrie::Operands;
    public:

        const Subscript _subscript;

        /**
         * Basic Constructor.
         * @param subscript Subscript that defines what the operator does.
         */
        explicit Result(const Subscript &subscript) : _subscript{subscript.optimized()} {}


        const NDMap<T> &getResult(const Operands &operands) {
            const std::vector<std::shared_ptr<Subscript>> &sub_subscripts = _subscript.getSubSubscripts();
            if (not sub_subscripts.empty()) {
                return CrossProduct<T>{_subscript}.getResult(operands);
            } else {
                return Einsum<T>{_subscript}.getResult(operands);
            }
        }
    };
};

#endif //SPARSETENSOR_EINSUM_OPERATOR_RESULT
