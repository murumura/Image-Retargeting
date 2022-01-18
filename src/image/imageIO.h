#ifndef IMAGEIO_H
#define IMAGEIO_H
#include <image/image.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
#include <filesystem>
namespace Image {

    bool hasExtension(std::string filename, std::string extension)
    {
        std::string ext = extension;
        if (!ext.empty() && ext[0] == '.')
            ext.erase(0, 1);
        std::string filenameExtension = std::filesystem::path(filename).extension();
        if (ext.size() > filenameExtension.size())
            return false;
        return std::equal(ext.rbegin(), ext.rend(), filenameExtension.rbegin(),
            [](char a, char b) { return std::tolower(a) == std::tolower(b); });
    }

    template <typename TColorDepth = uint8_t>
    void loadPNG(const std::string& filename, int numChannels, Eigen::Tensor<TColorDepth, 3, Eigen::RowMajor>& image)
    {
        if (!hasExtension(filename, "png"))
            throw std::invalid_argument("Expect file with png(PNG) extension, got: " + filename);

        // STB image will automatically return numChannels
        int width, height, channel;
        unsigned char* buf = stbi_load(filename.c_str(), &width, &height, &channel, numChannels);
        if (buf) {
            image.resize(height, width, numChannels);
            for (int r = 0; r < image.dimension(0); ++r) {
                for (int c = 0; c < image.dimension(1); ++c) {
                    for (int d = 0; d < image.dimension(2); ++d) {
                        int i = r * width * numChannels + c * numChannels + d;
                        image(r, c, d) = static_cast<TColorDepth>((buf)[i]);
                    }
                }
            }
            stbi_image_free(buf);
        }
        else
            throw std::invalid_argument("Could not load from file path " + filename);
    }

    template <typename TColorDepth>
    auto loadPNG(const std::string& filename, std::size_t numChannels)
    {
        Eigen::Tensor<TColorDepth, 3, Eigen::RowMajor> image;
        loadPNG(filename, numChannels, image);
        return image;
    }

    template <typename TColorDepth, int numChannels>
    void savePNG(std::string filename, const Eigen::Tensor<TColorDepth, numChannels, Eigen::RowMajor>& image)
    {
        if (!hasExtension(filename, "png"))
            filename += ".png";
        int height = image.dimension(0);
        int width = image.dimension(1);
        int channels = image.dimension(2);
        TColorDepth* buf = new TColorDepth[height * width * channels];
        for (Index r = 0; r < image.dimension(0); ++r) {
            for (Index c = 0; c < image.dimension(1); ++c) {
                for (Index d = 0; d < image.dimension(2); ++d) {
                    Index i = r * width * channels + c * channels + d;
                    buf[i] = static_cast<TColorDepth>(image(r, c, d));
                }
            }
        }
        stbi_write_png(filename.c_str(), image.dimension(1), image.dimension(0), image.dimension(2), buf, 0);
        delete[] buf;
    }

} // namespace Image
#endif
