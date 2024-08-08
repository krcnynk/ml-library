#include "ml_framework/tensor.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>

namespace ml_framework
{

  Tensor::Tensor(const std::vector<int> &shape) : _shape(shape)
  {
    _data.resize(std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>()));
    // shape std vector int container, begin and end addresses, start value 1, a function object
  }

  Tensor::Tensor(const std::vector<int> &shape, const std::vector<float> &data) : _shape(shape), _data(data)
  // copy constructor is being called in member initializer
  {
    if (data.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()))
    {
      throw std::invalid_argument("Data size does not match shape.");
    }
  }

  const std::vector<int> &Tensor::shape() const
  {
    return _shape;
  }
  
  const std::vector<float> &Tensor::data() const
  {
    return _data;
  }

  float *Tensor::data()
  {
    return _data.data();
  }

  Tensor Tensor::operator+(const Tensor &other)
  {
    if (_shape != other._shape)
    {
      throw std::invalid_argument("Shapes do not match.");
    }
    Tensor result(_shape);
    std::transform(other._data.cbegin(), other._data.cend(), this->_data.cbegin(), result._data.begin(), std::plus<float>());
    return result;

    // for (size_t i = 0; i < data_.size(); ++i)
    // {
    //   result.data_[i] = data_[i] + other.data_[i];
    // }
    // return result;
  }

  Tensor Tensor::operator*(const Tensor &other)
  {
    //  TODO
    if (_shape != other._shape)
    {
      throw std::invalid_argument("Shapes do not match.");
    }
    Tensor result(_shape);
    std::transform(other._data.cbegin(), other._data.cend(), this->_data.cbegin(), result._data.begin(), std::multiplies<float>());
    return result;


    // if (_shape != other._shape)
    // {
    //   throw std::invalid_argument("Shapes do not match.");
    // }
    // Tensor result(_shape);
    // for (size_t i = 0; i < data_.size(); ++i)
    // {
    //   result.data_[i] = data_[i] * other.data_[i];
    // }
    // return result;
  }

}