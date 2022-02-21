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
        float SegmentK{500.0};
        int MinSize{30};
        float MergePercent {0.0001};
        float MergeColorDist {20.0};
        bool SaveSegment{true};
        int DistC{3};
        int SimilarK{64};
        int NumScale{3};
        int ScaleU{4};
        float Overlap{0.5};
        bool SaveSaliency{true};
        bool SaveScaledSaliency{true};
        int newH{400};
        int newW{400};
    };

    auto parser = CommndLineParser<myOpts>::create({
        {"--InputImage", &myOpts::InputImage, "Input image location"},
        {"--Sigma", &myOpts::Sigma, "Gaussian blur sigma"},
        {"--SegmentK", &myOpts::SegmentK, "Segment threshold"},
        {"--MinSize", &myOpts::MinSize, "Segment area threshold"},
        {"--MergePercent", &myOpts::MergePercent, "Additional merge area threshold (in percentage) threshold"},
        {"--MergeColorDist", &myOpts::MergeColorDist, "Additional merge color distance threshold"},
        {"--SaveSegment", &myOpts::SaveSegment, "Whether to save segmentation (with each pixel represent by their original color) result."},
        {"--DistC", &myOpts::DistC, "Scale variable of position distance"},
        {"--SimilarK", &myOpts::SimilarK, "K most similar patches"},
        {"--NumScale", &myOpts::NumScale, "Number of Patches Scale"},
        {"--ScaleU", &myOpts::ScaleU, "Patches Scale value"},
        {"--Overlap", &myOpts::Overlap, "Patches Overlap percents"},
        {"--SaveScaledSaliency", &myOpts::SaveScaledSaliency, "Whether to save saliency result of each scale."},
        {"--SaveSaliency", &myOpts::SaveSaliency, "Whether to save saliency result."},
        {"--newH", &myOpts::newH, "Resizing Height."},
        {"--newW", &myOpts::newW, "Resizing Weight."}
    });

    auto args = parser->parse(argc, argv);
    
    std::shared_ptr<Image::GraphSegmentation> graphSeg = \
        Image::createGraphSegmentation(
        args.Sigma, 
        args.SegmentK, 
        args.MinSize, 
        args.MergePercent, 
        args.MergeColorDist
    );

    // Load input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input = \
        Image::loadPNG<uint8_t>(args.InputImage, 3);
    
    // Store segment patch information
    std::vector<Image::Patch> patches;

    // Store segment ID of each pixel
    Eigen::Tensor<int, 3, Eigen::RowMajor> segMapping(input.dimension(0), input.dimension(1), 1);

    // Store segment result of input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> segResult;

    graphSeg->processImage(input, patches, segMapping, segResult);

    if (args.SaveSegment)
        Image::savePNG("./segmentation", segResult);
    
    auto caSaliency = Image::createContextAwareSaliency(
        args.DistC, 
        args.SimilarK, 
        args.NumScale,
        args.ScaleU,
        args.SaveScaledSaliency
    );

    // Store saliency map of input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> saliencyMap;

    Eigen::Tensor<float, 3, Eigen::RowMajor> significanceMap;

    caSaliency->processImage(input, saliencyMap);

    if (args.SaveScaledSaliency)
        Image::savePNG("./saliency", saliencyMap);
    
    Image::Wrapping::assignSignificance(saliencyMap, segMapping, significanceMap, patches);
}
