#include <args.h>
#include <image/image.h>
#include <image/imageIO.h>
#include <image/saliency.h>
#include <image/segmentation.h>
#include <image/wrapping.h>
int main(int argc, const char* argv[])
{
    struct myOpts {
        std::string InputImage{"./datasets/maple.png"};
        float Sigma{0.5};
        float SegmentK{500.0};
        int MinSize{100};
        float MergePercent{0.0001};
        float MergeColorDist{20.0};
        bool SaveSegment{true};
        int DistC{3};
        int SimilarK{64};
        int NumScale{3};
        int ScaleU{6};
        bool SaveSaliency{true};
        bool SaveScaledSaliency{true};
        int newH{400};
        int newW{400};
        float Alpha{0.8f};
        int QuadSize{10};
        float WeightDST{1.0f};
        float WeightDLT{1.0f};
        float WeightDOR{10.0f};
    };

    auto parser = CommndLineParser<myOpts>::create({{"--InputImage", &myOpts::InputImage, "Input image location"},
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
        {"--SaveScaledSaliency", &myOpts::SaveScaledSaliency, "Whether to save saliency result of each scale."},
        {"--SaveSaliency", &myOpts::SaveSaliency, "Whether to save saliency result."},
        {"--newH", &myOpts::newH, "Resizing Height."},
        {"--newW", &myOpts::newW, "Resizing Weight."},
        {"--Alpha", &myOpts::Alpha, "Weighting factor for the energy terms Dst and Dlt."},
        {"--QuadSize", &myOpts::QuadSize, "Height/Width of each quad to be deformed."},
        {"--WeightDST", &myOpts::WeightDST, "Weight factor for avoid over-deformation on patches with low significant"},
        {"--WeightDLT", &myOpts::WeightDLT, "Weight factor for avoid over-deformation on patches with low significant"},
        {"--WeightDOR", &myOpts::WeightDOR, "Weight factor for avoid skew artifacts."}});

    auto args = parser->parse(argc, argv);

    std::shared_ptr<Image::GraphSegmentation> graphSeg = Image::createGraphSegmentation(
        args.Sigma,
        args.SegmentK,
        args.MinSize,
        args.MergePercent,
        args.MergeColorDist);

    // Load input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> input = Image::loadPNG<uint8_t>(args.InputImage, 3);

    // Store segment patch information
    std::vector<Image::Patch> patches;

    // Store segment ID of each pixel
    Eigen::Tensor<int, 3, Eigen::RowMajor> segMapping(input.dimension(0), input.dimension(1), 1);

    // Store segment result of input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> segResult;

    graphSeg->processImage(input, patches, segMapping, segResult);

    if (args.SaveSegment)
        Image::savePNG("./segmentation" + std::to_string(args.SegmentK) + "-" + std::to_string(args.MinSize), segResult);

    auto caSaliency = Image::createContextAwareSaliency(
        args.DistC,
        args.SimilarK,
        args.NumScale,
        args.ScaleU,
        args.SaveScaledSaliency);

    // Store saliency map of input image
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> saliencyMap;

    Eigen::Tensor<float, 3, Eigen::RowMajor> significanceMap;

    caSaliency->processImage(input, saliencyMap);

    if (args.SaveSaliency)
        Image::savePNG("./saliency", saliencyMap);

    Image::Wrapping::assignSignificance(saliencyMap, segMapping, significanceMap, patches);

    if (args.SaveSaliency)
        Image::savePNG<uint8_t, 3>("./significance", significanceMap.cast<uint8_t>());

    // Retargeting results
    Eigen::Tensor<uint8_t, 3, Eigen::RowMajor> resizedImage;

    auto wrapping = Image::createWrapping(
        args.newH,
        args.newW,
        args.Alpha,
        args.QuadSize,
        args.WeightDST,
        args.WeightDLT,
        args.WeightDOR);

    wrapping->reconstructImage<uint8_t>(input, segMapping, patches, resizedImage);
}
