#ifndef __QCNN__H__
#define __QCNN__H__
#include "Numcpp.hpp"
#include <vector>

namespace qcnn
{
    template <typename T>
    struct qcnn_layer
    {
        np::Numcpp<T> layer_matrix;
        
    };

    /*
    QCNN: Provide a neural network implemented with matrices and processors.
    */
    class qcnn
    {
    private:
    public:
        qcnn(/* args */);
        ~qcnn();
    };
} // namespace qcnn

#endif //!__QCNN__H__