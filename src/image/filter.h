#include <image/image.h>
#include <image/pad_op.h>
namespace Image {
    namespace Internal {
        struct traits : public traits<XprType> {
            typedef typename XprType::Scalar Scalar;
            typedef traits<XprType> XprTraits;
            typedef typename XprTraits::StorageKind StorageKind;
            typedef typename XprTraits::Index Index;
            typedef typename XprType::Nested Nested;
            typedef typename remove_reference<Nested>::type _Nested;
            static constexpr int NumDimensions = XprTraits::NumDimensions;
            static constexpr int Layout = XprTraits::Layout;
        };
    } // namespace Internal
} // namespace Image
