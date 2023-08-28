#include <iostream>
#include <fstream>
#include <vector>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

using namespace std::chrono;



__global__
void computeDistanceKernel(double* datasetMean, double* queryMean, double* datasetMedian, double* queryMedian, double* datasetStdDev,
    double* queryStdDev, double* datasetHuMoments, double* queryHuMoments, double* datasetHistogram, double* queryHistogram,
    double* distances, int datasetSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < datasetSize)
    {
        double distance = 0.0;

        distance += std::pow(datasetMean[index] - queryMean[0], 2);
        distance += std::pow(datasetMedian[index] - queryMedian[0], 2);
        distance += std::pow(datasetStdDev[index] - queryStdDev[0], 2);

        for (int i = 0; i < 7; ++i)
        {
            distance += std::pow(datasetHuMoments[index * 7 + i] - queryHuMoments[i], 2);
        }

        for (int i = 0; i < 256; ++i)
        {
            distance += std::pow(datasetHistogram[index * 256 + i] - queryHistogram[i], 2);
        }

        distances[index] = std::sqrt(distance);
    }
}

// Function to resize an image to a given size
cv::Mat resizeImage(const cv::Mat& image, int width, int height)
{
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(width, height));
    return resizedImage;
}

// Function to compute the mean value of an image
double computeMean(const cv::Mat& image)
{
    cv::Scalar meanVal = cv::mean(image);
    return meanVal[0];
}

// Function to compute the median value of an image
double computeMedian(const cv::Mat& image)
{
    cv::Mat sortedImage;
    cv::sort(image, sortedImage, cv::SORT_EVERY_COLUMN | cv::SORT_ASCENDING);
    int totalPixels = image.rows * image.cols;

    double medianValue;
    if (totalPixels % 2 == 0)
    {
        int index1 = (totalPixels / 2) - 1;
        int index2 = index1 + 1;
        medianValue = (sortedImage.at<uchar>(index1) + sortedImage.at<uchar>(index2)) / 2.0;
    }
    else
    {
        int index = (totalPixels / 2);
        medianValue = sortedImage.at<uchar>(index);
    }

    return medianValue;
}

// Function to compute the histogram of an image
cv::Mat computeHistogram(const cv::Mat& image)
{
    int numBins = 256; // Number of bins for the histogram
    int histSize[] = { numBins };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    int channels[] = { 0 }; // Compute histogram only for the first channel (grayscale image)

    cv::Mat histogram;
    cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 1, histSize, ranges);

    return histogram;
}

// Function to compute the standard deviation of an image
double computeStandardDeviation(const cv::Mat& image)
{
    double mean = computeMean(image);
    cv::Scalar stddev;
    cv::meanStdDev(image, cv::noArray(), stddev);
    return stddev[0];
}

// Function to compute the Hu moments of an image
cv::Mat computeHuMoments(const cv::Mat& image)
{
    cv::Mat moments;
    cv::HuMoments(cv::moments(image), moments);
    return moments;
}

// Function to load the dataset of images
void loadDataset(const std::string& datasetPath, std::vector<cv::Mat>& datasetImages)
{
    datasetImages.clear();

    // Open the directory
    DIR* dir = opendir(datasetPath.c_str());
    if (dir == nullptr)
    {
        std::cerr << "Error opening directory: " << datasetPath << std::endl;
        return;
    }

    // Read the directory entries
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;

        // Skip directories and hidden files
        if (entry->d_type == DT_DIR || filename[0] == '.')
            continue;

        std::string imagePath = datasetPath + "/" + filename;

        // Load the image and add it to the dataset
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty())
        {
            std::cerr << "Error loading image: " << imagePath << std::endl;
            continue;
        }

        datasetImages.push_back(image);
    }

    // Close the directory
    closedir(dir);
}

struct DistanceIndexPair {
    double distance;
    int index;
};

int main()
{
    auto start = high_resolution_clock::now();
    std::string datasetPath = "./dataset/images/"; // Replace with the path to your dataset directory
    std::string queryImagePath = "000000124442.jpg"; // Replace with the path to your query image

    // Load the dataset images
    std::vector<cv::Mat> datasetImages;
    loadDataset(datasetPath, datasetImages);

    // Load the query image
    cv::Mat queryImage = cv::imread(queryImagePath, cv::IMREAD_COLOR);
    if (queryImage.empty())
    {
        std::cerr << "Error loading query image: " << queryImagePath << std::endl;
        return 1;
    }

    // Convert the query image to grayscale
    cv::Mat queryImageGray;
    cv::cvtColor(queryImage, queryImageGray, cv::COLOR_BGR2GRAY);

    // Resize the query image
    cv::Mat resizedQueryImage = resizeImage(queryImageGray, 128, 128);

    // Compute the features for the query image
    double queryMean = computeMean(resizedQueryImage);
    double queryMedian = computeMedian(resizedQueryImage);
    double queryStdDev = computeStandardDeviation(resizedQueryImage);
    cv::Mat queryHuMoments = computeHuMoments(resizedQueryImage);
    cv::Mat queryHistogram = computeHistogram(resizedQueryImage);

    // Allocate GPU memory for dataset features and distances
    int datasetSize = datasetImages.size();
    double* deviceDatasetMean;
    double* deviceQueryMean;
    double* deviceDatasetMedian;
    double* deviceQueryMedian;
    double* deviceDatasetStdDev;
    double* deviceQueryStdDev;
    double* deviceDatasetHuMoments;
    double* deviceQueryHuMoments;
    double* deviceDatasetHistogram;
    double* deviceQueryHistogram;
    double* deviceDistances;
    cudaMalloc(&deviceDatasetMean, datasetSize * sizeof(double));
    cudaMalloc(&deviceQueryMean, sizeof(double));
    cudaMalloc(&deviceDatasetMedian, datasetSize * sizeof(double));
    cudaMalloc(&deviceQueryMedian, sizeof(double));
    cudaMalloc(&deviceDatasetStdDev, datasetSize * sizeof(double));
    cudaMalloc(&deviceQueryStdDev, sizeof(double));
    cudaMalloc(&deviceDatasetHuMoments, datasetSize * 7 * sizeof(double));
    cudaMalloc(&deviceQueryHuMoments, 7 * sizeof(double));
    cudaMalloc(&deviceDatasetHistogram, datasetSize * 256 * sizeof(double));
    cudaMalloc(&deviceQueryHistogram, 256 * sizeof(double));
    cudaMalloc(&deviceDistances, datasetSize * sizeof(double));
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "GPU Memory allocation time: " << duration.count() << " milliseconds" << std::endl;

    auto start1 = high_resolution_clock::now();

    // Copy dataset features to GPU memory
    for (int i = 0; i < datasetSize; ++i)
    {
        cv::Mat resizedImage;
        cv::Mat imageGray;
        cv::cvtColor(datasetImages[i], imageGray, cv::COLOR_BGR2GRAY);
        resizedImage = resizeImage(imageGray, 128, 128);

        double imageMean = computeMean(resizedImage);
        double imageMedian = computeMedian(resizedImage);
        double imageStdDev = computeStandardDeviation(resizedImage);
        cv::Mat imageHuMoments = computeHuMoments(resizedImage);
        cv::Mat imageHistogram = computeHistogram(resizedImage);

        cudaMemcpy(deviceDatasetMean + i, &imageMean, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceDatasetMedian + i, &imageMedian, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceDatasetStdDev + i, &imageStdDev, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceDatasetHuMoments + (i * 7), imageHuMoments.ptr<double>(), 7 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceDatasetHistogram + (i * 256), imageHistogram.ptr<double>(), 256 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // Copy query features to GPU memory
    cudaMemcpy(deviceQueryMean, &queryMean, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueryMedian, &queryMedian, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueryStdDev, &queryStdDev, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueryHuMoments, queryHuMoments.ptr<double>(), 7 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceQueryHistogram, queryHistogram.ptr<double>(), 256 * sizeof(double), cudaMemcpyHostToDevice);

    // Set the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int numBlocks = (datasetSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to compute distances
    computeDistanceKernel<<<numBlocks, threadsPerBlock>>>(deviceDatasetMean, deviceQueryMean, deviceDatasetMedian,
        deviceQueryMedian, deviceDatasetStdDev, deviceQueryStdDev, deviceDatasetHuMoments, deviceQueryHuMoments,
        deviceDatasetHistogram, deviceQueryHistogram, deviceDistances, datasetSize);

    // Copy the distances from GPU memory to the host
    double* distances = new double[datasetSize];
    cudaMemcpy(distances, deviceDistances, datasetSize * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<DistanceIndexPair> distanceIndexPairs;
    for (int i = 0; i < datasetSize; ++i) {
        DistanceIndexPair pair;
        pair.distance = distances[i];
        pair.index = i;
        distanceIndexPairs.push_back(pair);
    }

    std::sort(distanceIndexPairs.begin(), distanceIndexPairs.end(), [](const DistanceIndexPair& a, const DistanceIndexPair& b) {
        return a.distance < b.distance;
    });

    std::cout << "20 lowest distances:" << std::endl;
    for (int i = 0; i < 20; ++i) {
        std::cout << "Index: " << distanceIndexPairs[i].index << ", Distance: " << distanceIndexPairs[i].distance << std::endl;
    }




    // Free GPU memory
    cudaFree(deviceDatasetMean);
    cudaFree(deviceQueryMean);
    cudaFree(deviceDatasetMedian);
    cudaFree(deviceQueryMedian);
    cudaFree(deviceDatasetStdDev);
    cudaFree(deviceQueryStdDev);
    cudaFree(deviceDatasetHuMoments);
    cudaFree(deviceQueryHuMoments);
    cudaFree(deviceDatasetHistogram);
    cudaFree(deviceQueryHistogram);
    cudaFree(deviceDistances);

    // Free host memory
    delete[] distances;

    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);
    std::cout << "Execution time: " << duration1.count() << " milliseconds" << std::endl;


    return 0;
}
