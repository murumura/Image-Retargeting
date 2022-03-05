#ifndef UTILS_H
#define UTILS_H
#include <stddef.h>

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
namespace Utils {
    namespace detail {
        // Trait to select overloads and return types for MakeUnique.
        template <typename T>
        struct MakeUniqueResult {
            using scalar = std::unique_ptr<T>;
        };
        template <typename T>
        struct MakeUniqueResult<T[]> {
            using array = std::unique_ptr<T[]>;
        };
        template <typename T, size_t N>
        struct MakeUniqueResult<T[N]> {
            using invalid = void;
        };

    } // namespace detail

    // Return lower-cased version of str.
    inline std::string stringToLower(const std::string& str)
    {
        std::string lowerStr = str;
        std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return lowerStr;
    }

    // Transfers ownership of a raw pointer to a std::unique_ptr of deduced type.
    // Note: Cannot wrap pointers to array of unknown bound (i.e. U(*)[]).
    template <typename T>
    std::unique_ptr<T> wrapUnique(T* ptr)
    {
        static_assert(!std::is_array<T>::value || std::extent<T>::value != 0,
            "types T[0] or T[] are unsupported");
        return std::unique_ptr<T>(ptr);
    }

    template <typename T, typename... Args>
    typename detail::MakeUniqueResult<T>::scalar makeUnique(Args&&... args)
    {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    // Overload for array of unknown bound.
    // The allocation of arrays needs to use the array form of new,
    // and cannot take element constructor arguments.
    template <typename T>
    typename detail::MakeUniqueResult<T>::array makeUnique(size_t n)
    {
        return std::unique_ptr<T>(new typename std::remove_extent<T>::type[n]());
    }

} // namespace Utils
#endif