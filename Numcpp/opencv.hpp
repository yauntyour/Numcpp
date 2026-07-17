#ifndef NUMCPP_OPENCV_HPP
#define NUMCPP_OPENCV_HPP

#include <opencv2/opencv.hpp>
#include "core.hpp"

namespace np
{
    template <typename T>
    Numcpp<T> fromCvMat(const cv::Mat &mat)
    {
        if (mat.empty()) throw std::invalid_argument("OpenCV matrix is empty.");
        Numcpp<T> result(mat.rows, mat.cols);
        for (int i = 0; i < mat.rows; i++)
        {
            const T *row_ptr = mat.ptr<T>(i);
            for (int j = 0; j < mat.cols; j++)
                result[i][j] = static_cast<T>(row_ptr[j]);
        }
        return result;
    }

    template <typename T>
    cv::Mat toCvMat(const Numcpp<T> &mat, int mat_type = -1)
    {
        if (mat.row == 0 || mat.col == 0) return cv::Mat();
        if (mat_type == -1)
        {
            if (std::is_same<T, float>::value)            mat_type = CV_32F;
            else if (std::is_same<T, double>::value)      mat_type = CV_64F;
            else if (std::is_same<T, uint8_t>::value)     mat_type = CV_8U;
            else if (std::is_same<T, int8_t>::value)      mat_type = CV_8S;
            else if (std::is_same<T, uint16_t>::value)    mat_type = CV_16U;
            else if (std::is_same<T, int16_t>::value)     mat_type = CV_16S;
            else if (std::is_same<T, int32_t>::value)     mat_type = CV_32S;
            else                                          mat_type = CV_64F;
        }
        cv::Mat result(static_cast<int>(mat.row), static_cast<int>(mat.col), mat_type);
        for (int i = 0; i < mat.row; i++)
        {
            T *row_ptr = result.ptr<T>(i);
            for (int j = 0; j < mat.col; j++)
                row_ptr[j] = static_cast<T>(mat[i][j]);
        }
        return result;
    }

} // namespace np

#endif // NUMCPP_OPENCV_HPP
