#ifndef ML_FRAMEWORK_TENSOR_H
#define ML_FRAMEWORK_TENSOR_H

#include <vector>
#include <memory>

namespace ml_framework
{

  template <typename T>
  struct AlignedAllocator;

  using AlignedFloatAllocator = AlignedAllocator<float>;
  using AlignedIntAllocator = AlignedAllocator<int>;

  template <typename T>
  struct AlignedAllocator
  {
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(std::size_t n)
    {
      void *ptr = nullptr;
      if (posix_memalign(&ptr, 64, n * sizeof(T)) != 0)
      {
        throw std::bad_alloc();
      }
      return static_cast<T *>(ptr);
    }

    void deallocate(T *ptr, std::size_t) noexcept
    {
      std::free(ptr);
    }
  };

  class Tensor
  {
  public:
    Tensor(const std::vector<int> &shape);
    Tensor(const std::vector<int> &shape, const std::vector<float> &data);

    const std::vector<int> &shape() const;
    const std::vector<float> &data() const;
    // will return read only, and can be called on const objects
    float *data();

    Tensor operator+(const Tensor &other);
    Tensor operator*(const Tensor &other);

  private:
    std::vector<int> _shape;
    std::vector<float> _data;
  };

}

#endif