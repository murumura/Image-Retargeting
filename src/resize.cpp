#include <image/image.h>
#include <image/imageIO.h>
#include <image/segmentation.h>
#include <image/wrapping.h>
#include <image/saliency.h>
#include <args.h>
int main(int argc, const char* argv[])
{
    struct myOpts
    {
        std::string InputImage{"./datasets/butterfly.png"};
        float Sigma{0.5};
        float SegmentK{300.0};
        int MinSize{100};
        float MergePercent {0.0001};
        float MergeColorDist {20.0};
        bool SaveSegment{true};
    };

    auto parser = CommndLineParser<myOpts>::create({
        {"--InputImage", &myOpts::InputImage, "Input image location"},
        {"--Sigma", &myOpts::Sigma, "Gaussian blur sigma"},
        {"--SegmentK", &myOpts::SegmentK, "Segment threshold"},
        {"--MinSize", &myOpts::MinSize, "Segment area threshold"},
        {"--MergePercent", &myOpts::MergePercent, "Additional merge area threshold (in percentage) threshold"},
        {"--MergeColorDist", &myOpts::MergeColorDist, "Additional merge color distance threshold"},
        {"--SaveSegment", &myOpts::SaveSegment, "Whether to save segmentation result."}
    });

    auto args = parser->parse(argc, argv);
    // clang-format off
    std::shared_ptr<Image::GraphSegmentation> graphSeg = \
        Image::createGraphSegmentation(
        args.Sigma, 
        args.SegmentK, 
        args.MinSize, 
        args.MergePercent, 
        args.MergeColorDist
    );
    // clang-format on 

    std::vector<Image::Patch> patches;

    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input = \
        Image::loadPNG<uint8_t>(args.InputImage, 3);

    // Store segment ID of each pixel
    Eigen::Tensor<int, 3, Eigen::RowMajor> segMapping(input.dimension(0), input.dimension(1), 1);

    // Store segment result of input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> segResult;

    graphSeg->processImage(input, patches, segMapping, segResult);

    if (args.SaveSegment)
        Image::savePNG("./segmentation", segResult);
    
}
