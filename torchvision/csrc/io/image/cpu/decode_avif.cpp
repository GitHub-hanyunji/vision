#include "decode_avif.h"

#if AVIF_FOUND
#include "avif/avif.h"
#endif // AVIF_FOUND

namespace vision {
namespace image {

#if !AVIF_FOUND
torch::Tensor decode_avif(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  TORCH_CHECK(
      false, "decode_avif: torchvision not compiled with libavif support");
}
#else

// This normally comes from avif_cxx.h, but it's not always present when
// installing libavif. So we just copy/paste it here.
struct UniquePtrDeleter {
  void operator()(avifDecoder* decoder) const {
    avifDecoderDestroy(decoder);
  }
};
using DecoderPtr = std::unique_ptr<avifDecoder, UniquePtrDeleter>;

torch::Tensor decode_avif(
    const torch::Tensor& encoded_data,
    ImageReadMode mode) {
  // This is based on
  // https://github.com/AOMediaCodec/libavif/blob/main/examples/avif_example_decode_memory.c
  // Refer there for more detail about what each function does, and which
  // structure/data is available after which call.

  TORCH_CHECK(encoded_data.is_contiguous(), "Input tensor must be contiguous.");
  TORCH_CHECK(
      encoded_data.dtype() == torch::kU8,
      "Input tensor must have uint8 data type, got ",
      encoded_data.dtype());
  TORCH_CHECK(
      encoded_data.dim() == 1,
      "Input tensor must be 1-dimensional, got ",
      encoded_data.dim(),
      " dims.");

  DecoderPtr decoder(avifDecoderCreate());
  TORCH_CHECK(decoder != nullptr, "Failed to create avif decoder.");

  auto result = AVIF_RESULT_UNKNOWN_ERROR;
  result = avifDecoderSetIOMemory(
      decoder.get(), encoded_data.data_ptr<uint8_t>(), encoded_data.numel());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderSetIOMemory failed:",
      avifResultToString(result));

  result = avifDecoderParse(decoder.get());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderParse failed: ",
      avifResultToString(result));
  TORCH_CHECK(
      decoder->imageCount == 1, "Avif file contains more than one image");

  result = avifDecoderNextImage(decoder.get());
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifDecoderNextImage failed:",
      avifResultToString(result));

  avifRGBImage rgb;
  memset(&rgb, 0, sizeof(rgb));
  avifRGBImageSetDefaults(&rgb, decoder->image);

  // images encoded as 10 or 12 bits will be decoded as uint16. The rest are
  // decoded as uint8.
  auto use_uint8 = (decoder->image->depth <= 8);
  rgb.depth = use_uint8 ? 8 : 16;

  if (mode != IMAGE_READ_MODE_UNCHANGED && mode != IMAGE_READ_MODE_RGB &&
      mode != IMAGE_READ_MODE_RGB_ALPHA) {
    // Other modes aren't supported, but we don't error or even warn because we
    // have generic entry points like decode_image which may support all modes,
    // it just depends on the underlying decoder.
    mode = IMAGE_READ_MODE_UNCHANGED;
  }

  // If return_rgb is false it means we return rgba - nothing else.
  auto return_rgb =
      (mode == IMAGE_READ_MODE_RGB ||
       (mode == IMAGE_READ_MODE_UNCHANGED && !decoder->alphaPresent));

  auto num_channels = return_rgb ? 3 : 4;
  rgb.format = return_rgb ? AVIF_RGB_FORMAT_RGB : AVIF_RGB_FORMAT_RGBA;
  rgb.ignoreAlpha = return_rgb ? AVIF_TRUE : AVIF_FALSE;

  auto out = torch::empty(
      {rgb.height, rgb.width, num_channels},
      use_uint8 ? torch::kUInt8 : at::kUInt16);
  rgb.pixels = (uint8_t*)out.data_ptr();
  rgb.rowBytes = rgb.width * avifRGBImagePixelSize(&rgb);

  result = avifImageYUVToRGB(decoder->image, &rgb);
  TORCH_CHECK(
      result == AVIF_RESULT_OK,
      "avifImageYUVToRGB failed: ",
      avifResultToString(result));

  return out.permute({2, 0, 1}); // return CHW, channels-last
}
#endif // AVIF_FOUND

} // namespace image
} // namespace vision
