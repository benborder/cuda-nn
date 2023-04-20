#include "mnist.h"

namespace
{
constexpr std::uint32_t kimage_code = 2051U;
constexpr std::uint32_t klabel_code = 2049U;
} // namespace

static uint32_t convert_big_to_little_endian_if_necessary(uint32_t data)
{
	if constexpr (std::endian::native == std::endian::little)
	{
		data = ((data & 0xFF000000) >> 24) | ((data & 0x00FF0000) >> 8) | ((data & 0x0000FF00) << 8) |
					 ((data & 0x000000FF) << 24);
	}
	return data;
}

static uint32_t read_uint32(std::istream& is)
{
	if (uint32_t word; is.read(reinterpret_cast<char*>(&word), sizeof(word)))
	{
		return convert_big_to_little_endian_if_necessary(word);
	}
	else { throw std::runtime_error("Read error"); }
}

MNISTDatasetLoader::MNISTDatasetLoader(std::filesystem::path path) : path_(path)
{
	if (!std::filesystem::exists(path_))
	{
		spdlog::error("The supplied path '{}' does not exist", path_.string());
		throw std::runtime_error("The supplied path does not exist");
	}
}

std::vector<MNISTObject> MNISTDatasetLoader::load_dataset(std::istream& images, std::istream& labels) const
{
	if (read_uint32(images) != kimage_code) { throw std::runtime_error("Bad image code"); }
	if (read_uint32(labels) != klabel_code) { throw std::runtime_error("Bad label code"); }

	const std::uint32_t cnt_images = read_uint32(images);
	const std::uint32_t cnt_labels = read_uint32(labels);
	if (cnt_images != cnt_labels) { throw std::runtime_error("Image/label counts do not match"); }
	spdlog::debug("{} images", cnt_images);

	const std::uint32_t rows = read_uint32(images);
	const std::uint32_t cols = read_uint32(images);
	if (rows != MNISTObject::rows || cols != MNISTObject::cols)
	{
		spdlog::error("Incorrect image rows {}!={} or cols {}!={}", rows, MNISTObject::rows, cols, MNISTObject::cols);
		throw std::runtime_error("Incorrect image rows/cols");
	}
	spdlog::debug("{} rows, {} cols", rows, cols);

	const std::uint32_t img_size = rows * cols;

	std::vector<MNISTObject> data_set;
	data_set.reserve(cnt_images);
	for (std::size_t i = 0; i < cnt_images; ++i)
	{
		MNISTObject obj;
		obj.index = static_cast<int>(i);
		obj.image.reserve(img_size);
		auto pixels = std::make_unique<std::uint8_t[]>(img_size);
		if (!images.read(reinterpret_cast<char*>(pixels.get()), img_size))
		{
			spdlog::error("Image {} read error", i);
			throw std::runtime_error("Image read error");
		}
		std::transform(pixels.get(), pixels.get() + img_size, std::back_inserter(obj.image), [](std::uint8_t pixel) {
			return static_cast<float>(pixel) / 255.0F;
		});
		if (!labels.read(reinterpret_cast<char*>(&obj.label), sizeof(obj.label)))
		{
			spdlog::error("Label {} read error", i);
			throw std::runtime_error("Label read error");
		}
		if (obj.label >= MNISTObject::nlabels)
		{
			spdlog::error("Bad label: {} for pair {}", obj.label, i);
			throw std::runtime_error("Bad label");
		}
		data_set.push_back(obj);
	}
	spdlog::info("Success! {} images read", cnt_images);
	return data_set;
}

std::vector<MNISTObject> MNISTDatasetLoader::load_train_set() const
{
	spdlog::info("Loading MNIST train dataset...");
	std::filesystem::path train_images_path = path_ / "train-images-idx3-ubyte";
	std::filesystem::path train_labels_path = path_ / "train-labels-idx1-ubyte";
	std::ifstream train_images_ifs{train_images_path, std::ios::binary};
	if (!train_images_ifs)
	{
		spdlog::error("Cannot open input file: {}", train_images_path.string());
		return {};
	}
	std::ifstream train_labels_ifs{train_labels_path, std::ios::binary};
	if (!train_labels_ifs)
	{
		spdlog::error("Cannot open input file: {}", train_labels_path.string());
		return {};
	}

	return load_dataset(train_images_ifs, train_labels_ifs);
}

std::vector<MNISTObject> MNISTDatasetLoader::load_test_set() const
{
	spdlog::info("Loading MNIST test dataset...");
	std::filesystem::path test_images_path = path_ / "t10k-images-idx3-ubyte";
	std::filesystem::path test_labels_path = path_ / "t10k-labels-idx1-ubyte";
	std::ifstream test_images_ifs{test_images_path, std::ios::binary};
	if (!test_images_ifs)
	{
		spdlog::error("Cannot open input file: {}", test_images_path.string());
		return {};
	}
	std::ifstream test_labels_ifs{test_labels_path, std::ios::binary};
	if (!test_labels_ifs)
	{
		spdlog::error("Cannot open input file: {}", test_labels_path.string());
		return {};
	}

	return load_dataset(test_images_ifs, test_labels_ifs);
}
